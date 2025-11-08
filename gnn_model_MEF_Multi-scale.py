# gnn_model_MEF_Multi-scale.py
import os, glob, math
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
import networkx as nx
from scipy.sparse import csgraph
import matplotlib.pyplot as plt

# ========== 配置 ==========
DATA_DIR = r"C:\Users\qgz23\Desktop\CarbonFactor\data\IEEE_39_Multi"   # 用原始字符串避免转义
#多尺度窗口配置（15min/步）
WINDOW_15MIN   = 80         #15min尺度：80步=20小时（短周期依赖）
WINDOW_1H   = 4             #1h尺度：4步=1小时（中周期依赖）
WINDOW_24H   = 96           #24h尺度：96步=24小时（长周期依赖）
MAX_WINDOW = max(WINDOW_15MIN, WINDOW_1H, WINDOW_24H)
RIDGE_ALPHA = 1e-1  # 岭回归正则化参数
LAPL_BETA   = 1e-2  # 拉普拉斯矩阵正则化参数
TRAIN_RATIO = 0.8     # 训练集比例
EPOCHS  = 20            #（训练轮数）
LR      = 1e-2           #（学习率） 提高一点更容易“破常数”
WEIGHT_DECAY = 1e-5  #（权重衰减） 防止过拟合
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR= "./outputs_st_multi_scale"
MODEL_MODE = 'st_moe'    # 可选: "gcn_only" / "stgnn"/"st_mod"
LAMBDA_SMOOTH = 1e-5    #拉普拉斯平滑系数（轻度约束）
os.makedirs(SAVE_DIR, exist_ok=True)
SEED=20250809; np.random.seed(SEED); torch.manual_seed(SEED)    #固定随机种子（实验可复现）

# ========== 读取基准与图 ==========
def load_base():
    bus0 = pd.read_csv(os.path.join(DATA_DIR,"bus_base.csv"), header=None)  #节点信息
    gen0 = pd.read_csv(os.path.join(DATA_DIR,"gen_base.csv"), header=None)  #发电机组信息
    br0  = pd.read_csv(os.path.join(DATA_DIR,"branch_base.csv"), header=None)   #支路信息
    return bus0, gen0, br0

def graph_from_branch(br0, bus0):
    BUS_I=0; F_BUS=0; T_BUS=1; XCOL=3    #列索引定义（适配IEEE39数据）
    bus_ids = bus0.iloc[:,BUS_I].astype(int).values    # 节点ID
    id2idx = {b:i for i,b in enumerate(bus_ids)}    #节点ID映射到索引
    f = br0.iloc[:,F_BUS].astype(int).map(id2idx).values    # 支路起点索引
    t = br0.iloc[:,T_BUS].astype(int).map(id2idx).values    # 支路终点索引
    # 注意：训练里我们先不使用 edge_weight，避免权重尺度把消息传递毁掉
    edge_index = np.vstack([np.concatenate([f,t]), np.concatenate([t,f])])  # 边索引（双向存储）

    G = nx.Graph(); N=len(bus_ids)
    G.add_nodes_from(range(N)); G.add_edges_from([(int(fi),int(ti)) for fi,ti in zip(f,t)])
    A = nx.to_scipy_sparse_array(G, nodelist=range(N), format="csr")    # 邻接矩阵（稀疏）
    L = csgraph.laplacian(A, normed=False).toarray()    # 拉普拉斯矩阵（平滑正则）
    deg = np.array([d for _,d in G.degree()], dtype=float)  # 节点度数（特征输入）
    return edge_index, L, deg, len(bus_ids), bus_ids

# ========== 读取时序 ==========
def list_time_steps():
    bus_files = sorted(glob.glob(os.path.join(DATA_DIR, "bus_t_*.csv")))    # 遍历所以节点时序文件
    steps=[]
    for bf in bus_files:
        t = int(os.path.basename(bf).split('_')[-1].split('.')[0])  # 提取时间步编号
        gf = os.path.join(DATA_DIR, f"gen_t_{t:05d}.csv")    # 对应发电机组时序文件
        rf = os.path.join(DATA_DIR, f"branch_t_{t:05d}.csv")    # 对应支路时序文件
        if os.path.exists(gf) and os.path.exists(rf):    # 确保三件套完整
            steps.append(t)
    return steps

def compute_co2_series_from_gen():
    """严格按 eg_used.csv + gen_t_*.csv 重算系统CO2（tCO2）"""
    eg_used = pd.read_csv(os.path.join(DATA_DIR,'eg_used.csv'), header=None).values.reshape(-1)  # 发电机碳排放系数（tCO2/MWh）
    steps = sorted(int(os.path.basename(p).split('_')[-1].split('.')[0])
                   for p in glob.glob(os.path.join(DATA_DIR, "gen_t_*.csv")))    # 所有发电机时序时间步
    co2_map = {}
    for t in steps:
        gen_t = pd.read_csv(os.path.join(DATA_DIR, f"gen_t_{t:05d}.csv"), header=None).values
        Pg = np.maximum(gen_t[:,1].astype(float), 0.0)    # 第2列：发电机有功功率（MW），取正（避免不合理数据）
        co2_map[t] = float(np.sum(eg_used * Pg))    # 计算系统总CO2排放量
    return co2_map

# ========== μ 标签反演 ==========
def invert_mu_window(P_hist, CO2_hist, L, t_idx, window=80, alpha=1e-1, beta=1e-2):
    """返回 μ_t 以及窗口拟合残差 rmse；包含自适应正则与窗口回退"""
    N = P_hist.shape[1]  # 节点数
    lo = max(0, t_idx - window + 1); hi = t_idx + 1      # 窗口边界
    X = P_hist[lo:hi] - P_hist[[t_idx]]  # 输入特征：窗口内节点功率相对于当前步的变化量[W,N]
    y = CO2_hist[lo:hi] - CO2_hist[t_idx]   # 目标变量：窗口内CO2排放量相对于当前步的变化量[W]
    m = np.isfinite(y)  # 过滤掉无效数据
    X = X[m]; y = y[m]
    if len(y) == 0:
        return np.zeros(N), np.nan  # 窗口内无有效数据，返回默认值
    # 构建正则化方程组
    XtX = X.T @ X; Xty = X.T @ y
    A = XtX + alpha*np.eye(N) + beta*L  # 正则化系数矩阵（加入L2正则+拉普拉斯正则）

    try: condA = np.linalg.cond(A)  # 条件数（判断是否病态）
    except: condA = 1e12
    fac_used = 1.0
    if condA > 1e8:
        fac_used = min(condA/1e8, 100.0)
        A = XtX + (alpha*fac_used)*np.eye(N) + beta*L
        # 窗口回退（缩短窗口避免工况切换导致数据混叠）
        if (hi-lo) > 40:
            X = X[-40:]; y = y[-40:]
            XtX = X.T @ X; Xty = X.T @ y
            A = XtX + (alpha*fac_used)*np.eye(N) + beta*L

    try:
        mu = np.linalg.solve(A, Xty)    # 解线性方程组
    except np.linalg.LinAlgError:
        mu = np.linalg.lstsq(A, Xty, rcond=None)[0]  # 最小二乘解

    y_hat = X @ mu   # 预测值
    rmse = float(np.sqrt(np.mean((y_hat - y)**2)))  # 窗口拟合残差
    return mu, rmse

# ========== 构建多尺度数据集 ==========
def build_dataset():
    bus0, gen0, br0 = load_base()
    edge_idx_np, L, deg, N, bus_ids = graph_from_branch(br0, bus0)
    PD_col=2    #节点功率列索引

    steps = list_time_steps()
    if len(steps)==0: raise RuntimeError("未发现 bus_t_*.csv 三件套。")

    # 加载时序数据并过滤无效数据
    P_hist=[]; valid_map=[]
    for t in steps:
        bus_t = pd.read_csv(os.path.join(DATA_DIR, f"bus_t_{t:05d}.csv"), header=None).values
        if not np.isfinite(bus_t).all(): continue    # 过滤无效数据
        P_hist.append(bus_t[:,PD_col].astype(float))    #提取节点功率
        valid_map.append(t)
    P_hist = np.vstack(P_hist)  # [T_valid, N]：有效时间步*节点数
    T_valid = P_hist.shape[0]

    # 构造节点特征：当前功率、功率变化、节点度数（3维特征）
    deg_vec = deg
    X_frames=[]
    for k in range(T_valid):
        pd_now = P_hist[k]
        pd_prev= P_hist[max(0,k-1)] #（避免k=0时取不到前一时刻功率）
        x = np.stack([pd_now, pd_now-pd_prev, deg_vec], axis=1)  # [N,3]
        X_frames.append(x)
    X_frames = np.asarray(X_frames)  # [T_valid, N, 3]

    # 特征标准化（仅用训练集统计量，避免数据泄露）
    split_tmp = int(TRAIN_RATIO*T_valid)
    flat_train = X_frames[:split_tmp].reshape(-1, X_frames.shape[-1])
    mu_feat = flat_train.mean(0); std_feat=flat_train.std(0)+1e-8    # 均值、标准差（+1e-8防止除零）
    X_frames = (X_frames - mu_feat)/std_feat

    # 加载并对齐CO2排放量时序
    co2_map = compute_co2_series_from_gen()
    co2_vec = np.array([co2_map.get(t, np.nan) for t in valid_map])

    # 加载全局工况特征并标准化
    g_list = []
    for t in valid_map:
        try:
            g_list.append(regime_feats_for_t(t))
        except:
            g_list.append(np.array([0.0, 0.0], dtype=float))    # 若无对应时序文件，用默认值代替
    g_arr = np.vstack(g_list)  # [T_valid, 2]
    mu_g = g_arr[:split_tmp].mean(0)
    std_g = g_arr[:split_tmp].std(0) + 1e-8
    g_arr = (g_arr - mu_g) / std_g

    # 组装多尺度PyG数据集（适配GNN训练）
    data_list=[]
    edge_index = torch.tensor(edge_idx_np, dtype=torch.long)
    for k in range(T_valid):
        if k < MAX_WINDOW-1: continue    # 跳过过早时间步（不足窗口长度）
        #反演当前步的μ标签
        mu_k, _ = invert_mu_window(P_hist, co2_vec, L, k, window=WINDOW_15MIN,
                                   alpha=RIDGE_ALPHA, beta=LAPL_BETA)  # [N]
        #提取多尺度时序特征序列
        seq_15min = X_frames[k-WINDOW_15MIN+1:k+1]  # [80,N,3]
        seq_1h = X_frames[k-WINDOW_1H+1:k+1]       # [4,N,3]
        seq_24h = X_frames[k-WINDOW_24H+1:k+1]      # [96,N,3]
        g_vec = torch.tensor(g_arr[k], dtype=torch.float)  # 全局工况特征[2]
        # 构建PyG数据对象
        data = Data(
            x_15min=torch.tensor(seq_15min, dtype=torch.float), #15min尺度输入
            x_1h=torch.tensor(seq_1h, dtype=torch.float),     #1h尺度输入
            x_24h=torch.tensor(seq_24h, dtype=torch.float),    #24h尺度输入
            y=torch.tensor(np.asarray(mu_k), dtype=torch.float).view(-1,1), #标签[N,1]
            edge_index=edge_index,    #图边索引（双向）
            g=g_vec  #全局工况特征[2]
        )
        data.num_nodes = N    # 节点数
        data_list.append(data)

    return data_list, N, bus_ids, mu_g, std_g


def regime_feats_for_t(t):
    """计算某时间步的全局工况特征（拥堵状态识别）"""
    # 你导出的 branch_t_* 里若没有潮流列，可先用 DC 角差/x 计算；若已有 PF 列直接用它
    BR = pd.read_csv(os.path.join(DATA_DIR, f"branch_t_{t:05d}.csv"), header=None).values
    RATE_A = 5   # 支路额定容量列索引（case39 默认第6列，0-based=5）
    # 若有潮流列 PF/|S| 请改成对应列；没有就退化：以 θ 差推流略过（不影响特征趋势）
    if BR.shape[1] > 13:
        PF_COL = 13 #潮流列索引
        flow = np.abs(BR[:, PF_COL].astype(float))  # 绝对值支路潮流
    else:
        flow = np.zeros(len(BR))  # 没有就用0占位，不会影响能学性的比较
    rate = np.maximum(BR[:, RATE_A].astype(float), 1e-6)    # 支路额定容量（避免除以0）
    loading = flow / rate    # 支路载荷（流量/容量）
    bind_ratio = float(np.mean(loading > 0.95))  # 绑定占比
    max_loading = float(np.max(loading))    # 最大载荷
    return np.array([bind_ratio, max_loading], dtype=float)


# ========== 模型 ==========
class PosEnc(nn.Module):
    """Transformer位置编码：为时序特征注入位置信息（解决Transformer无顺序感知问题）"""
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # 位置编码矩阵 [max_len, d_model]
        pos = torch.arange(0, max_len).unsqueeze(1).float()  # 位置索引 [max_len, 1]
        # 频率因子（按维度衰减，避免位置信息混淆）
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)  # 偶数位置用sin
        pe[:, 1::2] = torch.cos(pos * div)  # 奇数位置用cos
        self.register_buffer('pe', pe)  # 注册为缓冲（不参与梯度更新）
    def forward(self, x):          # x: [N, W, d]（节点数*时间步数*特征维度）
        W = x.size(1)
        return x + self.pe[:W].unsqueeze(0)  # 加上位置信息

class STMoE(nn.Module):
    """多尺度时空混合专家模型：融合15min/1h/24h多尺度时序特征+空间依赖+工况自适应"""
    def __init__(self, in_dim, d_model=64, gcn_hid=128, K=3, dropout=0.1):
        super().__init__()
        #多尺度时序编码器（独立参数，避免尺度干扰）
        self.temporal_15min = TemporalEncoder(in_dim=in_dim, d_model=d_model, nhead=4,
                                        num_layers=2, dim_ff=4*d_model, dropout=dropout)    #短周期（15min）
        self.temporal_1h = TemporalEncoder(in_dim=in_dim, d_model=d_model, nhead=4,
                                        num_layers=2, dim_ff=4*d_model, dropout=dropout)    #中周期（1h）
        self.temporal_24h = TemporalEncoder(in_dim=in_dim, d_model=d_model, nhead=4,
                                             num_layers=3, dim_ff=4*d_model, dropout=dropout)    #长周期（24h，三层增强建模）
        self.proj_last = nn.Linear(in_dim, d_model)     #最后一帧特征维度映射
        self.fuse = nn.Linear(7*d_model, gcn_hid)   #多尺度特征融合（6个时序表征+1个最后一帧特征=7*d_model）
        self.g1 = GCNConv(gcn_hid, gcn_hid)    # GCN层1：捕捉空间依赖
        self.g2 = GCNConv(gcn_hid, gcn_hid)      # GCN层2：增强空间表征
        self.experts = nn.ModuleList([nn.Linear(gcn_hid, 1) for _ in range(K)])  # k个专家网络（适配不同工况）
        self.gate_lin = nn.Linear(gcn_hid + 2, K)    # 门控网络（输入空间特征+全局工况特征，输出k个专家权重）
        self.drop = dropout

    def forward(self, data):
        #多尺度时序编码
        x_last = data.x_15min[-1]    # 最后一帧特征[N,3]
        h_last_15, h_mean_15 = self.temporal_15min(data.x_15min)    #15min尺度特征[N,d_model]*2
        h_last_1h, h_mean_1h = self.temporal_1h(data.x_1h)    #1h尺度特征[N,d_model]*2
        h_last_24h, h_mean_24h = self.temporal_24h(data.x_24h)  #24h尺度特征[N,d_model]*2
        last_proj = self.proj_last(x_last)    # 最后一帧特征映射[N,d_model]
        # 融合多尺度特征
        z = torch.cat([h_last_15, h_mean_15,
                       h_last_1h, h_mean_1h,
                       h_last_24h, h_mean_24h,
                       last_proj], dim=-1)    # [N,7*d_model]
        z = F.relu(self.fuse(z))    # 融合为空间特征输入[N,gcn_hid]
        # GCN捕捉空间依赖
        h = F.relu(self.g1(z, data.edge_index))      # 空间表征1[N,gcn_hid]
        h = F.dropout(h, p=self.drop, training=self.training)    # dropout防止过拟合
        h = F.relu(self.g2(h, data.edge_index))          # 空间表征2[N,gcn_hid]

        # 门控网络（工况自适应分配专家权重）
        g_b = data.g.view(1, -1).repeat(h.size(0), 1)    # 全局工况特征广播到每个节点[N,2]
        gate_w = F.softmax(self.gate_lin(torch.cat([h, g_b], dim=-1)), dim=-1)  # 专家权重（softmax归一化）[N,K]

        # 混合专家预测
        outs = torch.stack([exp(h) for exp in self.experts], dim=-1)            # 每个专家输出[N,1,K]
        y = (outs.squeeze(1) * gate_w).sum(dim=-1, keepdim=True)                # 加权求和（权重自适应）[N,1]
        return y


class TemporalEncoder(nn.Module):
    """时序编码器：基于Transformer捕捉时序依赖，输出最后一步特征和平均特征"""
    def __init__(self, in_dim, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)    # 输入特征维度映射（3 -> d_model）
        #Transformer编码器层（批量优先、先归一化）
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers,enable_nested_tensor=False)    # 堆叠多层编码器
        self.pe = PosEnc(d_model)    # 位置编码
        self.ln = nn.LayerNorm(d_model)    # 层归一化（稳定训练）

    def forward(self, x_seq):      # x_seq: [W, N, F]
        x_seq = x_seq.permute(1, 0, 2)        # 维度转换[N, W, F]（适配Transformer批量优先）
        h = self.inp(x_seq)                   # 特征维度映射[N, W, d_model]
        h = self.pe(h)                       # 注入位置信息
        h = self.enc(h)                       # Transformer编码（捕捉时序依赖）
        h = self.ln(h)                       # 层归一化
        h_last = h[:, -1, :]                  # 最后一步特征[N, d_model]（保留近期信息）
        h_mean = h.mean(dim=1)                # 序列平均特征[N, d_model]（保留全局趋势）
        return h_last, h_mean

class STGNN(nn.Module):
    """单尺度版本，未修改"""
    def __init__(self, in_dim, d_model=64, gcn_hid=128, dropout=0.1):
        super().__init__()
        self.temporal = TemporalEncoder(in_dim=in_dim, d_model=d_model, nhead=4,
                                        num_layers=2, dim_ff=4*d_model, dropout=dropout)
        self.proj_last = nn.Linear(in_dim, d_model)
        self.fuse = nn.Linear(3*d_model, gcn_hid)
        self.g1 = GCNConv(gcn_hid, gcn_hid)
        self.g2 = GCNConv(gcn_hid, gcn_hid)
        self.out = nn.Linear(gcn_hid, 1)
        self.drop = dropout
    def forward(self, data):
        x_last = data.x_15min[-1]                                # [N, F]
        h_last, h_mean = self.temporal(data.x_15min)             # [N,d], [N,d]
        last_proj = self.proj_last(x_last)                 # [N,d]
        z = torch.cat([h_last, h_mean, last_proj], dim=-1) # [N,3d]
        z = F.relu(self.fuse(z))                           # [N,H]
        h = F.relu(self.g1(z, data.edge_index))            # ⚠ 不传 edge_weight
        h = F.dropout(h, p=self.drop, training=self.training)
        h = F.relu(self.g2(h, data.edge_index))
        y = self.out(h)
        return y

class GCNOnly(nn.Module):
    def __init__(self, in_dim, hid=128, drop=0.0):
        super().__init__()
        self.g1 = GCNConv(in_dim, hid)
        self.g2 = GCNConv(hid, hid)
        self.out = nn.Linear(hid, 1)
        self.drop = drop
    def forward(self, data):
        x = data.x_15min[-1]  # 仅用最后一帧 [N,F]
        h = F.relu(self.g1(x, data.edge_index))
        h = F.dropout(h, p=self.drop, training=self.training)
        h = F.relu(self.g2(h, data.edge_index))
        return self.out(h)

def lap_smooth_loss(y_pred, edge_index):
    """拉普拉斯平滑损失：约束相邻节点预测值相近（符合电力系统空间关联性）"""
    i,j = edge_index    # 边的起始节点和终止节点索引
    return ((y_pred[i]-y_pred[j])**2).mean()    #相邻节点预测值差异的均值

def eval_flat_and_plot(model, valid_loader, mu_y, std_y, save_dir, use_calibrator=None):
    """模型评估：计算多维度指标+绘制时序对比图"""
    import numpy as np, matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, r2_score

    model.eval()    # 评估模式（dropout等不启用）
    y_all, p_all = [], []
    with torch.no_grad():    # 禁用梯度计算（节省内存+加速）
        for data in valid_loader:
            data = data.to(DEVICE)
            p_std = model(data).cpu().numpy().reshape(-1)   # 模型预测值（标准化后）
            y_std = data.y.cpu().numpy().reshape(-1)        # 真实值（标准化后）
            # 反标准化（还原为真实单位）
            p = p_std * std_y + mu_y
            y = y_std * std_y + mu_y
            # 可选：应用校准器（提升预测准确性）
            if use_calibrator is not None:
                p = use_calibrator(p)
            y_all.append(y); p_all.append(p)

    # 压平为向量（时间*节点）
    y_vec = np.concatenate(y_all)   # 真实值向量[T_valid*N]
    p_vec = np.concatenate(p_all)   # 模型预测值向量[T_valid*N]

    #计算多维度评估指标
    # 传统指标（MAE/RMSE/R2）
    mae  = float(mean_absolute_error(y_vec, p_vec))
    rmse = float(np.sqrt(np.mean((y_vec - p_vec) ** 2)))
    r2   = float(r2_score(y_vec, p_vec))

    # 稳健比例指标(对零/负值不敏感)
    abs_y = np.abs(y_vec)
    abs_err = np.abs(p_vec - y_vec)
    wmape = float(abs_err.sum() / (abs_y.sum() + 1e-12))    # 加权平均绝对百分比误差
    smape = float(np.mean(2.0 * abs_err / (np.abs(p_vec) + abs_y + 1e-12)))  # 对称平均百分比误差
    # 归一化指标（分位数范围更稳健）
    yr = np.percentile(y_vec, 95) - np.percentile(y_vec, 5) + 1e-12 #95%分位数-5%分位数
    nmae = mae  / yr     # 归一化MAE
    nrmse = rmse / yr    # 归一化RMSE

    print(f"[FLAT] MAE={mae:.6f}  RMSE={rmse:.6f}  R2={r2:.4f}")
    print(f"[FLAT] WMAPE={wmape*100:.2f}%  sMAPE={smape*100:.2f}%  NMAE={nmae:.4f}  NRMSE={nrmse:.4f}")

    # 绘制真实值 vs 模型预测值曲线
    plt.figure(figsize=(10,4))
    idx = np.arange(len(y_vec))
    p_range = np.arange(1,300)  #绘制前299个点（避免图过于拥挤）
    plt.plot(idx[p_range], y_vec[p_range], label="True", linewidth=1.2, alpha=0.9)
    plt.plot(idx[p_range], p_vec[p_range], label="Pred", linewidth=1.2, alpha=0.9)
    plt.xlabel("Flattened index (time × node)")
    plt.ylabel("MCF (unscaled)")
    plt.title("True vs Pred (flattened)")
    plt.legend(); plt.tight_layout()
    path = os.path.join(save_dir, "curve_true_pred_flat.png")
    plt.savefig(path, dpi=160); plt.close()
    print(f"曲线已保存: {path}")

    return dict(MAE=mae, RMSE=rmse, R2=r2, WMAPE=wmape, sMAPE=smape, NMAE=nmae, NRMSE=nrmse)



# ========== 训练与评估 ==========
def train_eval():
    # 数据准备
    data_list, N, bus_ids, mu_g, std_g = build_dataset()    # 加载多尺度数据集
    T = len(data_list); assert T > 1, "样本太少"

    # 标签分布统计（辅助参数调整）
    Y_all = np.stack([d.y.numpy().flatten() for d in data_list])
    print("μ_all: min/mean/max/std =", Y_all.min(), Y_all.mean(), Y_all.max(), Y_all.std())
    print("share(|μ|<1e-6) =", np.mean(np.abs(Y_all) < 1e-6))

    # 时间切分训练集与验证集（按时间顺序，避免数据泄露）
    split = int(TRAIN_RATIO * T)
    train_set = data_list[:split]
    valid_set = data_list[split:]
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)    # 训练集shuffle
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)    # 验证集不shuffle（保持时间顺序）

    # 训练集统计量标准化标签
    Y_train = np.stack([d.y.numpy().flatten() for d in train_set])
    mu_y = float(Y_train.mean()); std_y = float(Y_train.std() + 1e-8)
    for d in data_list:
        d.y = (d.y - mu_y) / std_y  # 标准化标签

    #基准模型训练（用于对比）
    # 常数基线（预测值=0）
    val_y_std = []
    with torch.no_grad():
        for d in valid_loader:
            val_y_std.append(d.y.numpy().flatten())
    val_y_std = np.concatenate(val_y_std)
    const0_mse = float(np.mean(val_y_std**2))
    print("Const0 baseline (std-space MSE) =", const0_mse)

    # Ridge线性基线（仅用最后一帧特征）
    X_tr = np.stack([d.x_15min[-1].numpy().reshape(-1) for d in train_set])
    Y_tr = np.stack([d.y.numpy().flatten()      for d in train_set])
    X_va = np.stack([d.x_15min[-1].numpy().reshape(-1) for d in valid_set])
    Y_va = np.stack([d.y.numpy().flatten()       for d in valid_set])
    ridge = Ridge(alpha=1.0).fit(X_tr, Y_tr)
    ridge_mse = float(np.mean((ridge.predict(X_va) - Y_va)**2))
    print("[Baseline] Ridge-lastframe Val MSE (std-space) =", ridge_mse)

    # 模型初始化
    in_dim = data_list[0].x_15min.shape[-1]  # 输入特征维度：3

    if MODEL_MODE == "gcn_only":
        model = GCNOnly(in_dim=in_dim, hid=128, drop=0.0).to(DEVICE)
    elif MODEL_MODE == "st_moe":
        model = STMoE(in_dim=in_dim, d_model=64, gcn_hid=128, K=3, dropout=0.1).to(DEVICE)
    else:
        model = STGNN(in_dim=in_dim, d_model=64, gcn_hid=96, dropout=0.1).to(DEVICE)

    crit = torch.nn.MSELoss()   #基础损失函数：MSE
    opt  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)   #优化器：Adam
    CLIP_NORM = 1.0  #梯度裁剪阈值（防止梯度爆炸）

    train_losses, val_mses = [], []  # 记录训练损失和验证MSE

    # 模型训练
    for ep in range(1, EPOCHS + 1):
        # 训练阶段
        model.train(); tr = 0.0
        for data in train_loader:
            data = data.to(DEVICE)
            pred = model(data)
            base = crit(pred, data.y)    # 基础MSE损失
            smooth = lap_smooth_loss(pred, data.edge_index)  # 拉普拉斯平滑损失
            loss = base + LAMBDA_SMOOTH * smooth     # 总损失(MSE+平滑)
            #梯度下降
            opt.zero_grad(); loss.backward()     # 梯度清零、反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)    #梯度裁剪
            opt.step()    # 更新参数
            tr += loss.item()
        tr /= max(1, len(train_loader))    # 训练损失（平均值）

        # 验证阶段
        model.eval(); va = 0.0
        with torch.no_grad():
            for data in valid_loader:
                data = data.to(DEVICE)
                p = model(data)
                va += F.mse_loss(p, data.y).item()  # 验证MSE损失
        va /= max(1, len(valid_loader))    # 验证MSE（平均值）
        train_losses.append(tr); val_mses.append(va)

        # 反标准化 RMSE 监控（真实单位）
        with torch.no_grad():
            y_s, p_s = [], []
            for data in valid_loader:
                data = data.to(DEVICE)
                p = model(data).cpu().numpy().flatten()
                y = data.y.cpu().numpy().flatten()
                y_s.append(y); p_s.append(p)
            y_un = np.concatenate(y_s)*std_y + mu_y
            p_un = np.concatenate(p_s)*std_y + mu_y
            rmse_un = float(np.sqrt(np.mean((y_un - p_un)**2)))
        print(f"Epoch {ep:03d} | train {tr:.4e} | val {va:.4e} | const0 {const0_mse:.4e} | RMSE(unscaled) {rmse_un:.4f}")

    # 分工况单调校准（提升预测准确性）
    model.eval()
    Ys, Ps, Cong = [], [], []    # 真实值、预测值、拥塞标记
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(DEVICE)
            p = model(data).cpu().numpy().flatten()
            y = data.y.cpu().numpy().flatten()
            y_un = y*std_y + mu_y
            p_un = p*std_y + mu_y
            # 反标准化回真实工况，bind_ratio>0.05 判拥塞
            bind_ratio_real = float(data.g[0].cpu().numpy()*std_g[0] + mu_g[0])
            cong = bind_ratio_real > 0.05
            Ys.append(y_un); Ps.append(p_un); Cong.append(np.full_like(y_un, cong, dtype=bool))
    y_un = np.concatenate(Ys); p_un = np.concatenate(Ps); cong_mask = np.concatenate(Cong)

    # 分别训练拥塞/非拥塞工况的Isotonic校准器（单调约束提升可靠性）
    if cong_mask.any():
        iso_c = IsotonicRegression(out_of_bounds='clip').fit(p_un[cong_mask],   y_un[cong_mask])    # 训练拥塞Isotonic校准器
    else:
        iso_c = None
    if (~cong_mask).any():
        iso_n = IsotonicRegression(out_of_bounds='clip').fit(p_un[~cong_mask], y_un[~cong_mask])    # 训练非拥塞Isotonic校准器
    else:
        iso_n = None

    #校准函数（按工况应用对应的校准器）
    def apply_calibration(p_un_vec, cong_vec):
        out = np.empty_like(p_un_vec)
        if iso_n is not None: out[~cong_vec] = iso_n.transform(p_un_vec[~cong_vec])
        else:                 out[~cong_vec] = p_un_vec[~cong_vec]
        if iso_c is not None: out[cong_vec]  = iso_c.transform(p_un_vec[cong_vec])
        else:                 out[cong_vec]  = p_un_vec[cong_vec]
        return out

    p_cal = apply_calibration(p_un, cong_mask)  # 校准后的预测值

    # 最终评估（未校准 & 校准）
    from sklearn.metrics import mean_absolute_error, r2_score
    MSE  = float(np.mean((y_un - p_un) ** 2));  RMSE = float(np.sqrt(MSE))
    MAE  = float(mean_absolute_error(y_un, p_un));  R2 = float(r2_score(y_un, p_un))

    MSEc = float(np.mean((y_un - p_cal) ** 2)); RMSEc = float(np.sqrt(MSEc))
    MAEc = float(mean_absolute_error(y_un, p_cal)); R2c = float(r2_score(y_un, p_cal))
    print(f"[VALID-unscaled]  RMSE={RMSE:.6f}  MAE={MAE:.6f}  R2={R2:.4f}")
    print(f"[CALIBRATED-by-regime] RMSE={RMSEc:.6f} MAE={MAEc:.6f} R2={R2c:.4f}")

    # 结果可视化与保存
    #训练/验证损失曲线
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='Train'); plt.plot(val_mses, label='Val MSE (std space)')
    plt.axhline(const0_mse, color='gray', linestyle='--', linewidth=1, label='Const0 baseline')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=160); plt.close()

    # 未校准散点图（真实值 vs 预测值）
    plt.figure(figsize=(6,6))
    plt.scatter(y_un, p_un, s=10, alpha=0.25)
    lo, hi = float(min(y_un.min(), p_un.min())), float(max(y_un.max(), p_un.max()))
    plt.plot([lo, hi], [lo, hi], 'r--', linewidth=2)    #理想预测线（y=x）
    plt.xlabel('True (unscaled)'); plt.ylabel('Pred (unscaled)')
    plt.title('Pred vs True (Validation)')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'scatter_true_pred.png'), dpi=160); plt.close()

    # 分工况校准后散点图
    plt.figure(figsize=(6,6))
    plt.scatter(y_un, p_cal, s=10, alpha=0.25)
    lo, hi = float(min(y_un.min(), p_cal.min())), float(max(y_un.max(), p_cal.max()))
    plt.plot([lo, hi], [lo, hi], 'r--', linewidth=2)
    plt.xlabel('True (unscaled)'); plt.ylabel('Pred (calibrated, by regime)')
    plt.title('Calibrated by Regime (Validation)')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'scatter_true_pred_calibrated.png'), dpi=160); plt.close()

    # 每节点 MAE（分析节点级预测性能）
    per_node_err = np.zeros(N); count = 0
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(DEVICE)
            p_un1 = model(data).cpu().numpy().flatten() * std_y + mu_y
            y_un1 = data.y.cpu().numpy().flatten() * std_y + mu_y
            per_node_err += np.abs(p_un1 - y_un1); count += 1
    per_node_mae = per_node_err / max(1, count)
    pd.DataFrame({'bus_id': bus_ids, 'mae': per_node_mae}).to_csv(
        os.path.join(SAVE_DIR, 'per_node_mae.csv'), index=False)

    #全局指标汇总
    pd.DataFrame([dict(MSE=MSE, RMSE=RMSE, MAE=MAE, R2=R2,
                       MSE_cal=MSEc, RMSE_cal=RMSEc, MAE_cal=MAEc, R2_cal=R2c)]).to_csv(
        os.path.join(SAVE_DIR, 'metrics_overall.csv'), index=False)

    print(f"保存到 {SAVE_DIR}；y标准化参数：mu_y={mu_y:.6f}, std_y={std_y:.6f}")
    _ = eval_flat_and_plot(model, valid_loader, mu_y, std_y, SAVE_DIR)


if __name__ == "__main__":
    train_eval()
