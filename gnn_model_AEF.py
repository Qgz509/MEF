c#########################
# 4. 定义 & 训练 GNN
#########################
# requirements: torch>=2.0, torch_geometric>=2.4
import torch
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch_geometric.data import TemporalData
from torch_geometric.nn import GATv2Conv
from model import CarbonGNN
from en_calculation import EN
import time

# ---------------- hyper‑parameters ----------------
num_cases = 4000
train_test_rate = 0.7  # 70% 训练集，30% 测试集
BATCH_SIZE = 2
HID = 128
LR = 5e-4
WEIGHT_DECAY = 5e-4
PATIENCE = 20
DROP_P = 0.2
CLIP = 1.0
EPOCHS = 20
data_dir = r"D://OneDrive//电碳-大数据中心//IEEE_code//data//"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()

# 读取数据
##
bus_features = []
branch_features = []
gen_features = []
pg_vecs = []
##
for i in range(1, num_cases + 1):
    bus_path = os.path.join(data_dir, f"bus_case_{i:02d}.csv")
    branch_path = os.path.join(data_dir, f"branch_case_{i:02d}.csv")
    gen_path = os.path.join(data_dir, f"gen_case_{i:02d}.csv")

    if not os.path.exists(bus_path) or not os.path.exists(branch_path):
        break  # 若文件不存在则跳过

    bus_df = pd.read_csv(bus_path, header=None)
    branch_df = pd.read_csv(branch_path, header=None)
    gen_df = pd.read_csv(gen_path, header=None)

    N = bus_df.shape[0]
    pg_vec = np.zeros(N, dtype=np.float32)
    for _, row in gen_df.iterrows():
        pg_vec[int(row[0]) - 1] = row[1]  # 0‑based index

    bus_feat = np.stack([
        bus_df.iloc[:, 7].to_numpy(),  # VM
        bus_df.iloc[:, 8].to_numpy(),  # VA
        pg_vec  # PG
    ], axis=1)  # [N,3]
    branch_feat = branch_df.iloc[:, [13]].to_numpy()  # 入节点+出节点 有功 PF + 无功 QF
    gen_feat = gen_df.iloc[:, [1]].to_numpy()  # 发电机节点+有功 PG
    branch_features.append(branch_feat)
    bus_features.append(bus_feat)
    pg_vecs.append(pg_vec)

edge_index = torch.tensor(branch_df.iloc[:, [0, 1]].to_numpy(), dtype=torch.long) - 1  # [2, E]
bus_features = torch.tensor(np.stack(bus_features),dtype=torch.float32)
branch_features = torch.tensor(np.stack(branch_features),dtype=torch.float32)

# ------------------ labels (EN) -------------------
# >>> Fill your existing EN() mapping here <<<
# mapping, OFFSET, etc. kept identical to original script
mapping = {
    "Generator": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Bus": [38, 30, 31, 32, 33, 34, 35, 36, 37, 29]
}
OFFSET = 1
key2branch = {
    (int(edge_index[i, 0]), int(edge_index[i,1])): i for i in range(edge_index.shape[0])
}
EG = torch.tensor([835, 650, 600, 835, 400, 835, 250, 200, 0, 835], dtype=torch.float)
K = len(mapping["Generator"]) # 10 台发电机
N = 39 # node

EN_function = EN(N, K, key2branch, OFFSET, mapping, EG)

labels = torch.stack([EN_function.build_EN(branch_features[t], torch.tensor(pg_vecs[t])[N-10:N]) for t in range(len(bus_features))])  # [T,N]

# ---------------- split + normalise --------------
T = bus_features.shape[0]

n_train = int(num_cases * train_test_rate)
n_test = num_cases - n_train
train_bus = bus_features[:n_train]
test_bus = bus_features[n_train:]

train_gen = gen_features[:n_train]
train_branch = branch_features[:n_train]

test_branch = branch_features[n_train:]
test_gen = gen_features[n_train:]

train_labels = labels[:n_train]
test_labels = labels[n_train:]
#
# normalize
bus_mean, bus_std = train_bus.mean([0,1]), train_bus.std([0,1]) + 1e-6
br_mean,  br_std  = train_branch.mean([0,1]), train_branch.std([0,1]) + 1e-6
lab_mean, lab_std = train_labels.mean(),          train_labels.std() + 1e-6

norm  = lambda x, m, s: (x - m) / s
inv   = lambda x, m, s: x * s + m

train_bus = norm(train_bus, bus_mean, bus_std)
train_branch = norm(train_branch, br_mean, br_std)
train_labels = norm(train_labels, lab_mean, lab_std)

test_bus  = norm(test_bus,  bus_mean, bus_std)
test_branch = norm(test_branch, br_mean, br_std)
test_labels   = norm(test_labels,  lab_mean, lab_std)

print("训练集 bus 特征维度：", train_bus.shape)
print("训练集 branch 特征维度：", train_branch.shape)
print("测试集 bus 特征维度：", test_bus.shape)
print("测试集 branch 特征维度：", test_branch.shape)
##########################
# 3. 构造 TemporalData
##########################
base_edge_index = edge_index.clone()
data_list_train = []
for t in range(n_train):
    data_list_train.append(
        TemporalData(
            x=train_bus[t],
            edge_index=(base_edge_index.clone()).long(),
            edge_attr=train_branch[t].clone(), y=train_labels[t]
        )
    )
data_list_test = []
for t in range(n_test):
    data_list_test.append(
        TemporalData(
            x=test_bus[t], edge_index=(base_edge_index.clone()).long(),
            edge_attr=test_branch[t].clone(), y=test_labels[t]
        )
    )
train_loader = DataLoader(data_list_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(data_list_test,   batch_size=BATCH_SIZE)
############ GATv2Conv 版本 ############
# ------------------ GNN model --------------------
# class CarbonGNN(nn.Module):
#     def __init__(self, in_nf, edge_nf, hid=HID, out_nf=1):
#         super().__init__()
#         self.fc_in = nn.Linear(in_nf, hid) if in_nf != hid else nn.Identity()
#         self.conv1 = GATv2Conv(hid, hid, edge_dim=edge_nf, heads=4, concat=False, dropout=DROP_P)
#         self.norm1 = nn.LayerNorm(hid)
#         self.conv2 = GATv2Conv(hid, hid, edge_dim=edge_nf, heads=4, concat=False, dropout=DROP_P)
#         self.norm2 = nn.LayerNorm(hid)
#         self.drop  = nn.Dropout(DROP_P)
#         self.out   = nn.Linear(hid, out_nf)
#
#     def forward(self, x, edge_index, edge_attr):
#         x = self.fc_in(x)
#         h = F.relu(self.conv1(x, edge_index, edge_attr))
#         h = self.norm1(h + x)           # residual + LN
#         h2 = F.relu(self.conv2(h, edge_index, edge_attr))
#         h = self.norm2(h2 + h)
#         h = self.drop(h)
#         return self.out(h).squeeze(-1)
# model = CarbonGNN(in_nf=train_bus.size(-1),edge_nf=train_branch.shape[2])
# model = model.to(device)
# optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
# scaler = GradScaler('cuda')
# loss_fn = nn.MSELoss()

########## spectral GNN 版本 ##########
###### ------------------ GNN model --------------------
model = CarbonGNN(in_nf=train_bus.size(-1),edge_nf=train_branch.shape[2], hid=HID, out_nf=1)
model = model.to(device)
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
scaler = GradScaler('cuda')
loss_fn = nn.MSELoss()
#

# ---------------- training loop ------------------
best_loss, patience = float('inf'), 0
for epoch in range(EPOCHS):
    # ---- train ----
    model.train(); running = 0
    for batch in train_loader:
        batch = batch.to(device)
        # print(batch.edge_index.shape)  # ➜ torch.Size([2, 1472])
        # print(batch.edge_attr.shape)  # ➜ torch.Size([1472, 2])
        optim.zero_grad(set_to_none=True)
        with autocast():
            pred = model(batch.x, batch.edge_index.T, batch.edge_attr)
            loss = loss_fn(pred, batch.y)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        scaler.step(optim); scaler.update()
        running += loss.item() * batch.num_graphs
    train_loss = running / len(train_loader.dataset)

    # ---- val ----
    model.eval(); running = 0
    with torch.no_grad(), autocast():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index.T, batch.edge_attr)
            running += loss_fn(pred, batch.y).item() * batch.num_graphs
    val_loss = running / len(test_loader.dataset)
    sched.step()

    print(f"Ep {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

    if val_loss + 1e-4 < best_loss:
        best_loss, patience = val_loss, 0
        torch.save(model.state_dict(), "best_gnn.pt")

# ---------------- evaluation ---------------------
model.load_state_dict(torch.load("best_gnn.pt"))
model.eval(); preds_list = []
with torch.no_grad(), autocast():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index.T, batch.edge_attr)
        preds_list.append(pred.cpu())
preds = torch.cat(preds_list, dim=0)
preds = inv(preds, lab_mean, lab_std)
true  = inv(test_labels.flatten(), lab_mean, lab_std)

plt.figure(figsize=(10,4))
plt.plot(true.numpy()[:100], label="True", linestyle="--", marker=".")
plt.plot(preds.numpy()[:100], label="Pred", alpha=0.7)
plt.legend();
title_str = f"Epochs:{EPOCHS} train_num:{n_train} batch_size:{BATCH_SIZE}"
plt.title(title_str)
plt.tight_layout(); plt.show()

print(f"Total time: {time.time()-start_time:.1f}s | Best val loss (std‑scaled) {best_loss:.4f}")
