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
from mef_getdata import mef_getdata

# ========== 配置 ==========
DATA_DIR = r"F:\\CarbonFactor\\data\\IEEE_39_Multi_totransformer\\"   # 用原始字符串避免转义
#多尺度窗口配置（15min/步）
WINDOW_15MIN   = 16         #4h尺度：1（短周期依赖）
WINDOW_1H   = 32             #8h尺度：4步=1小时（中周期依赖）
WINDOW_24H   = 96           #24h尺度：96步=24小时（长周期依赖）
MAX_WINDOW = max(WINDOW_15MIN, WINDOW_1H, WINDOW_24H)
RIDGE_ALPHA = 1e-1  # 岭回归正则化参数
LAPL_BETA   = 1e-2  # 拉普拉斯矩阵正则化参数
TRAIN_RATIO = 0.8     # 训练集比例
EPOCHS  = 1           #（训练轮数）
LR      = 1e-2           #（学习率） 提高一点更容易“破常数”
WEIGHT_DECAY = 1e-5  #（权重衰减） 防止过拟合
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR= "./outputs_st_multi_scale"
MODEL_MODE = 'st_moe'    # 可选: "gcn_only" / "stgnn"/"st_mod"
LAMBDA_SMOOTH = 1e-5    #拉普拉斯平滑系数（轻度约束）
os.makedirs(SAVE_DIR, exist_ok=True)
SEED=20250809; np.random.seed(SEED); torch.manual_seed(SEED)    #固定随机种子（实验可复现）



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
        self.scale_gate = nn.Linear(3 * d_model + 2, 3)  # 输入=三个尺度摘要+全局g(2维)，输出3个尺度权重
        self.fuse = nn.Linear(d_model + d_model, gcn_hid)  # 改：融合后的尺度向量(d_model) + last_proj(d_model)
        # self.fuse = nn.Linear(7*d_model, gcn_hid)   #多尺度特征融合（6个时序表征+1个最后一帧特征=7*d_model）
        self.g1 = GCNConv(gcn_hid, gcn_hid)    # GCN层1：捕捉空间依赖
        self.g2 = GCNConv(gcn_hid, gcn_hid)      # GCN层2：增强空间表征

        def _make_expert():
            return nn.Sequential(
                nn.Linear(gcn_hid, gcn_hid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gcn_hid, 1)
            )

        self.experts = nn.ModuleList([_make_expert() for _ in range(K)])
        # self.experts = nn.ModuleList([nn.Linear(gcn_hid, 1) for _ in range(K)])  # k个专家网络（适配不同工况）
        self.gate_lin = nn.Linear(gcn_hid + gcn_hid + 2, K)  # 输入: z(融合后) + h(GCN后) + g

        self.drop = dropout


    def forward(self, data):
        #多尺度时序编码
        x_last = data.x_15min[-1]    # 最后一帧特征[N,3]
        h_last_15, h_attn_15, h_mean_15 = self.temporal_15min(data.x_15min)
        h_last_1h, h_attn_1h, h_mean_1h = self.temporal_1h(data.x_1h)
        h_last_24h, h_attn_24h, h_mean_24h = self.temporal_24h(data.x_24h)

        last_proj = self.proj_last(x_last)    # 最后一帧特征映射[N,d_model]
        # 融合多尺度特征
        # 每个尺度用 (last + attn) 做摘要（最小且有效）
        s15 = 0.5 * (h_last_15 + h_attn_15)  # [N,d]
        s1h = 0.5 * (h_last_1h + h_attn_1h)  # [N,d]
        s24 = 0.5 * (h_last_24h + h_attn_24h)  # [N,d]

        # 生成尺度权重：用全局g + 3个尺度摘要的“节点均值”（更稳）
        g_b = data.g.view(1, -1).repeat(s15.size(0), 1)  # [N,2]
        s_mean = torch.cat([s15, s1h, s24], dim=-1)  # [N,3d]
        scale_logits = self.scale_gate(torch.cat([s_mean, g_b], dim=-1))  # [N,3]
        scale_w = torch.softmax(scale_logits, dim=-1)  # [N,3]

        # 加权融合成一个尺度向量
        s = (scale_w[:, [0]] * s15 +
             scale_w[:, [1]] * s1h +
             scale_w[:, [2]] * s24)  # [N,d]

        # 再拼上最后一帧映射（保留你原来的 last_proj 思路）
        z = torch.cat([s, last_proj], dim=-1)  # [N,2d]
        z = F.relu(self.fuse(z))  # [N,gcn_hid]

        # GCN捕捉空间依赖
        h = F.relu(self.g1(z, data.edge_index, data.edge_weight))
        h = F.dropout(h, p=self.drop, training=self.training)
        h = F.relu(self.g2(h, data.edge_index, data.edge_weight))

        # 门控网络（工况自适应分配专家权重）
        g_b = data.g.view(1, -1).repeat(h.size(0), 1)    # 全局工况特征广播到每个节点[N,2]
        gate_in = torch.cat([z, h, g_b], dim=-1)  # [N, 2*gcn_hid + 2]
        gate_w = F.softmax(self.gate_lin(gate_in), dim=-1)
        # 混合专家预测
        outs = torch.stack([exp(h) for exp in self.experts], dim=-1)            # 每个专家输出[N,1,K]
        y = (outs.squeeze(1) * gate_w).sum(dim=-1, keepdim=True)                # 加权求和（权重自适应）[N,1]
        return y

class AttnPool(nn.Module):
    """单头 attention pooling：让模型学‘哪些时刻更重要’"""
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.scorer = nn.Linear(d_model, 1, bias=False)

    def forward(self, h):  # h: [N, W, d]
        s = self.scorer(torch.tanh(self.proj(h))).squeeze(-1)  # [N, W]
        w = torch.softmax(s, dim=1).unsqueeze(-1)              # [N, W, 1]
        z = (w * h).sum(dim=1)                                 # [N, d]
        return z


class TemporalEncoder(nn.Module):
    """时序编码器：Transformer + (last/attn/mean) 三种摘要"""
    def __init__(self, in_dim, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.pe = PosEnc(d_model)
        self.ln = nn.LayerNorm(d_model)
        self.pool = AttnPool(d_model)

    def forward(self, x_seq):  # x_seq: [W, N, F]
        x_seq = x_seq.permute(1, 0, 2)   # [N, W, F]
        h = self.inp(x_seq)              # [N, W, d]
        h = self.pe(h)
        h = self.enc(h)
        h = self.ln(h)

        h_last = h[:, -1, :]             # [N, d]
        h_attn = self.pool(h)            # [N, d]
        h_mean = h.mean(dim=1)           # [N, d]
        return h_last, h_attn, h_mean


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
        h = F.relu(self.g1(x, data.edge_index, data.edge_weight))
        h = F.dropout(h, p=self.drop, training=self.training)
        h = F.relu(self.g2(h, data.edge_index, data.edge_weight))
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
    getData = mef_getdata(DATA_DIR)
    data_list, N, bus_ids, mu_g, std_g = getData.build_dataset()    # 加载多尺度数据集
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
        for it, data in enumerate(train_loader):
            opt.zero_grad(set_to_none=True);
            data = data.to(DEVICE)
            pred = model(data)
            base = crit(pred, data.y)    # 基础MSE损失
            # smooth = lap_smooth_loss(pred, data.edge_index)  # 拉普拉斯平滑损失
            # loss = base + LAMBDA_SMOOTH * smooth     # 总损失(MSE+平滑)
            loss = base
            w0 = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            #梯度下降
            loss.backward()     # 梯度清零、反向传播
            # 统计全局梯度，而不是 next(...)
            tot = 0.0
            cnt = 0
            maxg = 0.0
            for n, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach().abs()
                tot += g.mean().item()
                cnt += 1
                maxg = max(maxg, g.max().item())
            # print("grad mean(all):", tot / max(1, cnt), "grad max(all):", maxg)

            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)    #梯度裁剪
            opt.step()    # 更新参数
            # 参数变化也建议统计所有参数
            delta = 0.0
            cnt = 0
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                delta += (p.detach() - w0[n]).abs().mean().item()
                cnt += 1
            # print("param delta(all):", delta / max(1, cnt))
            #
            # print("pred mean/std:", pred.detach().mean().item(), pred.detach().std().item())

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
