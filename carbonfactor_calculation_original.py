# requirements: torch>=2.0, torch_geometric>=2.4
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import TemporalData
from torch_geometric.nn import GCNConv, GATv2Conv
import re
########################
# edge_index.csv 在读取后直接转换成 0-based edge_index。
#
# 节点特征、边特征均按时间步展开再重塑，得到 [T,N,F]、[T,E,F] 的张量。
#
# build_EN 函数按文献公式逐时刻生成 节点真实碳因子 作为监督标签。
#
# 模型示例使用 GCN + GATv2，可以替换为 GraphSAGE, GIN 等；只需输出节点标量。
#
# 训练循环对 全部 1001 个时刻 独立监督，可改造成 torch_geometric_temporal 的时序批处理。
########################################
##########################
# 1. 读入节点 & 边特征
##########################
path = "G://王艺霖//史博文潮流数据//"
excel = pd.read_excel(path + "80-AC1-30-A0-2.36.xlsx", sheet_name="Sheet1")
edge_df = pd.read_csv(path + "ieee39_edge_index.csv")
edge_index = torch.tensor(edge_df.values.T, dtype=torch.long)  # [2,E]

# rebuild topology
# ------------- 已有数据 --------------
# excel      : pd.DataFrame           （潮流表，含 Pij/Pji 列）
# edge_df    : 两列 [source_node, target_node] 对应 edge_index
# edge_index : torch.tensor shape [2, E]

# ------------- 1) 解析支路两端节点 --------------
p_pij = re.compile(r"Pij\(p\.u\.\)-?([A-Za-z0-9\+]+)-BUS-(\d+)")
p_pji = re.compile(r"Pji\(p\.u\.\)-?([A-Za-z0-9\+]+)-BUS-(\d+)")

branch_dict = {}          # {branch_id: {"from": bus1, "to": bus2}}

for col in excel.columns:
    if not isinstance(col, str):          # 跳过非字符串列
        continue
    m1, m2 = p_pij.match(col), p_pji.match(col)
    if m1:                                # Pij ⇒ “from” 端
        bid, bus = m1.group(1), int(m1.group(2))
        branch_dict.setdefault(bid, {})["from"] = bus
        branch_dict[bid]["pij_col"] = col
    elif m2:                              # Pji ⇒ “to” 端
        bid, bus = m2.group(1), int(m2.group(2))
        branch_dict.setdefault(bid, {})["to"] = bus
        branch_dict[bid]["pji_col"] = col

    # ------------- 2) 构造 “无向键 → 支路编号” --------------
key2branch = {
    tuple(sorted((d["from"], d["to"]))): bid
    for bid, d in branch_dict.items()
    if {"from", "to"} <= d.keys()         # 只保留两端都解析到的
}
u0, v0 = edge_df.iloc[0]["source_node"], edge_df.iloc[0]["target_node"]
OFFSET = 0
if (tuple(sorted((u0, v0))) not in key2branch) and \
   (tuple(sorted((u0+1, v0+1))) in key2branch):
    OFFSET = 1
print(f"[INFO] 使用节点偏移 OFFSET = {OFFSET}")

# ---------- 抽取支路编号 + 有功功率 ----------
branch_id_list = []
p_active_list = []
p_active_list_ji = []
unmatched = []

for u, v in zip(edge_df["source_node"], edge_df["target_node"]):
    bus_u, bus_v = u + OFFSET, v + OFFSET
    key = tuple(sorted((bus_u, bus_v)))

    bid = key2branch.get(key)
    if bid is None:
        unmatched.append(key)
        branch_id_list.append("NA")
        p_active_list.append(np.nan)
        continue

    info = branch_dict[bid]

    # 判断方向，选 Pij
    if (info["from"], info["to"]) == (bus_u, bus_v):
        p_val = excel[info["pij_col"]].to_numpy(dtype=float)   # 只取第 0 行；多时刻换成 .to_numpy()
    elif (info["from"], info["to"]) == (bus_v, bus_u):
        p_val = excel[info["pij_col"]].to_numpy(dtype=float)

    # 判断方向，选 Pji
    if (info["from"], info["to"]) == (bus_u, bus_v):
        p_val_ji = excel[info["pji_col"]].to_numpy(dtype=float)   # 只取第 0 行；多时刻换成 .to_numpy()
    elif (info["from"], info["to"]) == (bus_v, bus_u):
        p_val_ji = excel[info["pji_col"]].to_numpy(dtype=float)
    branch_id_list.append(bid)
    p_active_list.append(p_val)
    p_active_list_ji.append(p_val_ji)

# ---------- 写回 edge_df ----------
edge_df["branch_id"] = branch_id_list
edge_df["P_active"]  = p_active_list
E = len(edge_df)  # 支路数
# -------- 2) 拼成 [E, T]，再转 Torch --------
edge_p_array = np.stack(p_active_list, axis=0).T     # shape [T,E]
edge_p_tensor = torch.tensor(edge_p_array, dtype=torch.float32)
# edge_p_tensor_t = edge_p_tensor.t()  # [T, E]  # 先转置再 reshape
# edge_p_tensor = edge_p_tensor.T  # [T, E]  # 先转置再 reshape
edge_p_array_ji = np.stack(p_active_list_ji, axis=0).T      # shape [T,E]
edge_p_tensor_ji = torch.tensor(edge_p_array_ji, dtype=torch.float32)
# edge_p_tensor_ji_t = edge_p_tensor_ji.t()  # [T, E]  # 先转置再 reshape

# 对比 edge_p_tensor 和 edge_p_tensor_ji
bias = edge_p_array + edge_p_array_ji

print("edge_p_tensor.shape =", edge_p_tensor.shape)  # (E, T)
# ------------- edge_index + edge_attr --------------
# 节点特征：电压+功角
node_cols = [c for c in excel.columns if c.startswith("U(")] + \
            [c for c in excel.columns if c.startswith("\u03b4(p.u.)-BUS")]
node_x = torch.tensor(excel[node_cols].values, dtype=torch.float)  # [T,N*F]
T, NF = node_x.shape  # NF = 39*2
N = 39
node_x = node_x.view(T, N, -1)  # [T,N,Fnode]

# 边特征：Pij
edge_attr = edge_p_tensor.view(T, E, -1)  # [T,E,Fedge]

##########################
# 2. 计算监督标签 EN(t)
##########################
# 发电机接入节点 & 碳因子 (kgCO2/MWh)
gen_nodes = torch.tensor([38, 30, 31, 32, 33, 34, 35, 36, 37, 29])  # 0-based
EG = torch.tensor([900, 650, 600, 500, 400, 300, 250, 200, 100, 10], dtype=torch.float)
K = len(gen_nodes)  # 10 台发电机

# 提取发电机出力 P_Gk(t)
gen_cols = [c for c in excel.columns if c.startswith("P(p.u.)-G")]
PGt = torch.tensor(excel[gen_cols].values, dtype=torch.float)  # [T,K]
mapping = {
    "Generator": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Bus":       [38, 30, 31, 32, 33, 34, 35, 36, 37, 29]
}

# 支路潮流矩阵 PB(t)  &  PN(t)  (忽略网损用直流潮流) -----------------

def build_EN(edge_attr_t, PGt_t, eps=1e-6):
    """可逆保障: 若 diag==0 → 加 ε ；不可逆仍报错则改用 pseudo-inverse"""
    # ---------- PB(t) ----------
    PB = torch.zeros((N, N))
    for idx, (u, v) in enumerate(edge_index.T):
        P = edge_attr_t[idx, 0]  # 取 Pij
        if P >= 0:
            PB[u, v] = P
        else:
            PB[v, u] = -P

    # ---------- PG(t) ----------
    PG = torch.zeros((K, N))
    PG[torch.arange(K), mapping["Bus"]] = PGt_t

    # ---------- PN(t) ----------
    PN_diag = PB.sum(0) + PG.sum(0)  # 仅入流 + 发电注入
    PN_diag = torch.where(PN_diag == 0, PN_diag + eps, PN_diag)  # 对角补 ε
    PN = torch.diag(PN_diag)

    A = PN - PB.T  # 理论中的可逆矩阵
    try:
        EN = torch.linalg.solve(A, PG.T @ EG)  # 比 inv @ b 数值更稳
    except torch._C._LinAlgError:
        # 仍奇异：退化到 Moore–Penrose 伪逆
        EN = torch.linalg.pinv(A) @ (PG.T @ EG)
    return EN  # [N]


labels = []
for t in range(T):
    EN_t = build_EN(edge_attr[t], PGt[t])  # edge_attr[t] 是 [E, Fedge]
    labels.append(EN_t)
labels = torch.stack(labels)  # [T, N]
# 保存labels
pd.DataFrame(labels.cpu().numpy()).to_csv("cf.csv", index=False)
##########################
# 3. 构造 TemporalData
##########################
data_list = []
for t in range(T):
    data_list.append(
        TemporalData(
            x=node_x[t], edge_index=edge_index,
            edge_attr=edge_attr[t], y=labels[t]
        )
    )


##########################
# 4. 定义 & 训练 GNN
##########################
class CarbonGNN(torch.nn.Module):
    def __init__(self, in_nf, hid=64, out_nf=1):
        super().__init__()
        self.gcn1 = GCNConv(in_nf, hid)
        self.gat = GATv2Conv(hid, hid, heads=4, concat=False)
        self.lin = torch.nn.Linear(hid, out_nf)

    def forward(self, x, edge_index, edge_attr):
        h = torch.relu(self.gcn1(x, edge_index))
        h = torch.relu(self.gat(h, edge_index))
        return self.lin(h).squeeze(-1)  # [N]


model = CarbonGNN(in_nf=node_x.size(-1))
optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(3):
    loss = 0
    for td in data_list:
        optim.zero_grad()
        pred = model(td.x, td.edge_index, td.edge_attr)
        l = loss_fn(pred, td.y)
        l.backward()
        optim.step()
        loss += l.item()
    if epoch % 20 == 0:
        print(f"epoch {epoch:03d} | loss={loss / len(data_list):.6f}")
