import os, glob, math
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import networkx as nx
from scipy.sparse import csgraph


class mef_getdata():
    def __init__(self, path, WINDOW_15MIN=1, WINDOW_1H=4, WINDOW_24H= 96,TRAIN_RATIO = 0.8,RIDGE_ALPHA = 1e-1,LAPL_BETA   = 1e-2 ):
        self.DATA_DIR = path
        self.br0 = None
        self.bus0 = None
        self.gen0 = None
        self.WINDOW_15MIN = WINDOW_15MIN
        self.WINDOW_1H = WINDOW_1H
        self.WINDOW_24H = WINDOW_24H
        self.TRAIN_RATIO = TRAIN_RATIO
        self.RIDGE_ALPHA = RIDGE_ALPHA
        self.LAPL_BETA = LAPL_BETA

    # ========== 读取基准与图 ==========

    def graph_from_branch(self):
        BUS_I = 0
        F_BUS = 0
        T_BUS = 1
        XCOL = 13
        self.bus0 = pd.read_csv(os.path.join(self.DATA_DIR, "bus_t_00001.csv"), header=None)  # 节点信息
        self.gen0 = pd.read_csv(os.path.join(self.DATA_DIR, "gen_t_00001.csv"), header=None)  # 发电机组信息
        self.br0 = pd.read_csv(os.path.join(self.DATA_DIR, "branch_t_00001.csv"), header=None)  # 支路信息
        bus_ids = self.bus0.iloc[:, BUS_I].astype(int).values
        id2idx = {b: i for i, b in enumerate(bus_ids)}
        f = self.br0.iloc[:, F_BUS].astype(int).map(id2idx).values
        t = self.br0.iloc[:, T_BUS].astype(int).map(id2idx).values
        edge_index = np.vstack([np.concatenate([f, t]), np.concatenate([t, f])])
        # PF = br0.iloc[:, XCOL].astype(float).values  # DC 潮流输出
        # edge_w = np.abs(PF)
        # edge_w = edge_w / (np.mean(edge_w) + 1e-6)
        # edge_weight = np.r_[edge_w, edge_w].astype(np.float32)  # (2E,)
        G = nx.Graph()
        N = len(bus_ids)
        G.add_nodes_from(range(N))
        G.add_edges_from([(int(fi), int(ti)) for fi, ti in zip(f, t)])
        A = nx.to_scipy_sparse_array(G, nodelist=range(N), format="csr")
        L = csgraph.laplacian(A, normed=False).toarray()
        deg = np.array([d for _, d in G.degree()], dtype=float)
        return edge_index, (f, t), L, deg, len(bus_ids), bus_ids


    def compute_co2_series_from_gen(self):
        """严格按 eg_used.csv + gen_t_*.csv 重算系统CO2（tCO2）"""
        eg_used = pd.read_csv(os.path.join(self.DATA_DIR, 'eg_used.csv'), header=None).values.reshape(
            -1)  # 发电机碳排放系数（tCO2/MWh）
        steps = self.list_time_steps()
        co2_map = {}
        for t in steps:
            gen_t = pd.read_csv(os.path.join(self.DATA_DIR, f"gen_t_{t:05d}.csv"), header=None).values
            Pg = np.maximum(gen_t[:, 1].astype(float), 0.0)  # 第2列：发电机有功功率（MW），取正（避免不合理数据）
            co2_map[t] = float(np.sum(eg_used * Pg))  # 计算系统总CO2排放量
        print("co2_map: ", co2_map)
        return co2_map


    # ========== 读取时序 ==========
    def list_time_steps(self):
        bus_files = glob.glob(os.path.join(self.DATA_DIR, "bus_t_*.csv"))  # 遍历所以节点时序文件
        steps = []
        for bf in bus_files:
            t = int(os.path.basename(bf).split('_')[-1].split('.')[0])  # 提取时间步编号
            gf = os.path.join(self.DATA_DIR, f"gen_t_{t:05d}.csv")  # 对应发电机组时序文件
            rf = os.path.join(self.DATA_DIR, f"branch_t_{t:05d}.csv")  # 对应支路时序文件
            mu = os.path.join(self.DATA_DIR, f"mu_t_{t:05d}.csv")
            if os.path.exists(gf) and os.path.exists(rf) and os.path.exists(mu):  # 确保三件套完整
                steps.append(t)
        return sorted(steps)


    def regime_feats_for_t(self,t):
        """计算某时间步的全局工况特征（拥堵状态识别）"""
        # 你导出的 branch_t_* 里若没有潮流列，可先用 DC 角差/x 计算；若已有 PF 列直接用它
        BR = pd.read_csv(os.path.join(self.DATA_DIR, f"branch_t_{t:05d}.csv"), header=None).values
        RATE_A = 5  # 支路额定容量列索引（case39 默认第6列，0-based=5）
        # 若有潮流列 PF/|S| 请改成对应列；没有就退化：以 θ 差推流略过（不影响特征趋势）
        if BR.shape[1] > 13:
            PF_COL = 13  # 潮流列索引
            flow = np.abs(BR[:, PF_COL].astype(float))  # 绝对值支路潮流
        else:
            flow = np.zeros(len(BR))  # 没有就用0占位，不会影响能学性的比较
        rate = np.maximum(BR[:, RATE_A].astype(float), 1e-6)  # 支路额定容量（避免除以0）
        loading = flow / rate  # 支路载荷（流量/容量）
        bind_ratio = float(np.mean(loading > 0.95))  # 绑定占比
        max_loading = float(np.max(loading))  # 最大载荷
        return np.array([bind_ratio, max_loading], dtype=float)



    # ========== μ 标签反演 ==========
    def invert_mu_window(self, P_hist, CO2_hist, L, t_idx, window=80, alpha=1e-1, beta=1e-2):
        """返回 μ_t 以及窗口拟合残差 rmse；包含自适应正则与窗口回退"""
        N = P_hist.shape[1]  # 节点数
        lo = max(0, t_idx - window);
        hi = t_idx  # 窗口边界
        X = P_hist[lo:hi] - P_hist[[hi - 1]]  # 输入特征：窗口内节点功率相对于当前步的变化量[W,N]
        y = CO2_hist[lo:hi] - CO2_hist[hi - 1]  # 目标变量：窗口内CO2排放量相对于当前步的变化量[W]
        m = np.isfinite(y)  # 过滤掉无效数据
        X = X[m]
        y = y[m]
        if len(y) == 0:
            return np.zeros(N), np.nan  # 窗口内无有效数据，返回默认值
        # 构建正则化方程组
        XtX = X.T @ X
        Xty = X.T @ y
        A = XtX + alpha * np.eye(N) + beta * L  # 正则化系数矩阵（加入L2正则+拉普拉斯正则）

        try:
            condA = np.linalg.cond(A)  # 条件数（判断是否病态）
        except:
            condA = 1e12
        fac_used = 1.0
        if condA > 1e8:
            fac_used = min(condA / 1e8, 100.0)
            A = XtX + (alpha * fac_used) * np.eye(N) + beta * L
            # 窗口回退（缩短窗口避免工况切换导致数据混叠）
            if (hi - lo) > 40:
                X = X[-40:]
                y = y[-40:]
                XtX = X.T @ X
                Xty = X.T @ y
                A = XtX + (alpha * fac_used) * np.eye(N) + beta * L

        try:
            mu = np.linalg.solve(A, Xty)  # 解线性方程组
        except np.linalg.LinAlgError:
            mu = np.linalg.lstsq(A, Xty, rcond=None)[0]  # 最小二乘解

        y_hat = X @ mu  # 预测值
        rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))  # 窗口拟合残差
        return mu, rmse

    # ========== 构建多尺度数据集 ==========
    def build_dataset(self):
        edge_index, from_to, L, deg, N, bus_ids = self.graph_from_branch()
        PD_col = 2  # 节点功率列索引
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        steps = self.list_time_steps()
        if len(steps) == 0: raise RuntimeError("未发现 bus_t_*.csv 三件套。")

        # 加载时序数据并过滤无效数据
        P_hist = []
        valid_map = []
        for t in steps:
            bus_t = pd.read_csv(os.path.join(self.DATA_DIR, f"bus_t_{t:05d}.csv"), header=None).values
            if not np.isfinite(bus_t).all(): continue  # 过滤无效数据
            P_hist.append(bus_t[:, PD_col].astype(float))  # 提取节点功率
            valid_map.append(t)
        P_hist = np.vstack(P_hist)  # [T_valid, N]：有效时间步*节点数
        T_valid = P_hist.shape[0]

        # 构造节点特征：当前功率、功率变化、节点度数（3维特征）
        deg_vec = deg
        X_frames = []
        for k in range(T_valid):
            pd_now = P_hist[k]
            pd_prev = P_hist[max(0, k - 1)]  # （避免k=0时取不到前一时刻功率）
            x = np.stack([pd_now, pd_now - pd_prev, deg_vec], axis=1)  # [N,3]
            X_frames.append(x)
        X_frames = np.asarray(X_frames)  # [T_valid, N, 3]

        # 特征标准化（仅用训练集统计量，避免数据泄露）
        split_tmp = int(self.TRAIN_RATIO * T_valid)
        flat_train = X_frames[:split_tmp].reshape(-1, X_frames.shape[-1])
        mu_feat = flat_train.mean(0);
        std_feat = flat_train.std(0) + 1e-8  # 均值、标准差（+1e-8防止除零）
        X_frames = (X_frames - mu_feat) / std_feat

        print("X_mean: ", mu_feat, "X_std: ", std_feat, "X_1: ", X_frames[1])
        # 加载并对齐CO2排放量时序
        co2_map = self.compute_co2_series_from_gen()
        co2_vec = np.array([co2_map.get(t, np.nan) for t in valid_map])

        # 加载全局工况特征并标准化
        g_list = []
        for t in valid_map:
            try:
                g_list.append(self.regime_feats_for_t(t))
            except:
                g_list.append(np.array([0.0, 0.0], dtype=float))  # 若无对应时序文件，用默认值代替
        g_arr = np.vstack(g_list)  # [T_valid, 2]
        mu_g = g_arr[:split_tmp].mean(0)
        std_g = g_arr[:split_tmp].std(0) + 1e-8
        g_arr = (g_arr - mu_g) / std_g

        print("gen_mean:", mu_g.mean, "gen_std:", std_g.mean, "g_arr_1: ", g_arr[0])

        # 组装多尺度PyG数据集（适配GNN训练）
        data_list = []
        MAX_WINDOW = max(self.WINDOW_15MIN, self.WINDOW_1H, self.WINDOW_24H)
        for k in range(T_valid):
            if k < MAX_WINDOW - 1: continue  # 跳过过早时间步（不足窗口长度）
            # 反演当前步的μ标签
            mu_k, _ = self.invert_mu_window(P_hist, co2_vec, L, k, window=MAX_WINDOW,
                                       alpha=self.RIDGE_ALPHA, beta=self.LAPL_BETA)  # [N]
            # 提取多尺度时序特征序列
            seq_15min = X_frames[k - self.WINDOW_15MIN + 1:k + 1]  # [80,N,3]
            # 取原始15min窗口
            raw_1h = X_frames[k - self.WINDOW_1H + 1:k + 1]  # [W1, N, F], W1=4
            raw_24h = X_frames[k - self.WINDOW_24H + 1:k + 1]  # [W24, N, F], W24=96

            # 1h尺度：把4个15min聚合成“1个小时token”（也可以保留多个小时token，这里给最小版）
            seq_1h = raw_1h.mean(axis=0, keepdims=True)  # [1, N, F]  （如果你希望长度=4就别聚合）

            # 24h尺度：按小时聚合，得到长度=24的低频序列
            # reshape: [96, N, F] -> [24, 4, N, F] -> mean over 4
            seq_24h = raw_24h.reshape(24, 4, raw_24h.shape[1], raw_24h.shape[2]).mean(axis=1)  # [24, N, F]

            g_vec = torch.tensor(g_arr[k], dtype=torch.float)  # 全局工况特征[2]
            # 构建PyG数据对象
            t_real = valid_map[k]
            BRt = pd.read_csv(os.path.join(self.DATA_DIR, f"branch_t_{t_real:05d}.csv"), header=None).values

            PF_COL = 13  # 你需要用一次 print(BRt.shape) + 核对列语义
            RATE_A = 5

            pf = np.abs(BRt[:, PF_COL].astype(float))
            rate = np.maximum(BRt[:, RATE_A].astype(float), 1e-6)

            edge_w = pf / rate  # 载荷率（推荐）
            edge_w = edge_w / (edge_w.mean() + 1e-6)
            edge_w = np.clip(edge_w, 0.0, 2.0)  # 防极端

            edge_weight = np.r_[edge_w, edge_w].astype(np.float32)  # (2E,)
            edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            data = Data(
                x_15min=torch.tensor(seq_15min, dtype=torch.float),  # 15min尺度输入
                x_1h=torch.tensor(seq_1h, dtype=torch.float),  # 1h尺度输入
                x_24h=torch.tensor(seq_24h, dtype=torch.float),  # 24h尺度输入
                y=torch.tensor(np.asarray(mu_k), dtype=torch.float).view(-1, 1),  # 标签[N,1]
                edge_index=edge_index,  # 图边索引（双向）
                edge_weight=edge_weight,
                g=g_vec  # 全局工况特征[2]
            )
            data.num_nodes = N  # 节点数
            data_list.append(data)
        print("edge_w mean/std/max:", edge_w.mean(), edge_w.std(), edge_w.max())

        return data_list, N, bus_ids, mu_g, std_g
