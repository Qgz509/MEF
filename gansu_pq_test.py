#########################
# 4. 定义 & 训练 GNN
#########################
# requirements: torch>=2.0, torch_geometric>=2.4
from datetime import timedelta

import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
#from en_calculation import EN
import torch.nn as nn
import time
from torch_geometric.data import TemporalData
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch.cuda.amp import autocast
from torch.amp import GradScaler

from model import *
from data_pipeline import *


start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ---------------- hyper‑parameters ----------------
num_cases = 10000
train_test_rate = 0.7  # 70% 训练集，30% 测试集
BATCH_SIZE = 64
EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 2e-4
PATIENCE = 20
CLIP = 1.0


if __name__ == '__main__':
    # 读取数据
    data_dir = "F://CarbonFactor//data//gansu_data//"

    ####################碳因子############################
    cei_path=os.path.join(data_dir,f"碳因子.xlsx")
    cei_dfs=pd.read_excel(cei_path,sheet_name=None, engine="openpyxl")
    cei_dfs.pop('0423', None)
    # 纵向拼接所有DataFrame（假设结构相同）
    cei_df = pd.concat(
        cei_dfs.values(),  # 提取所有DataFrame
        axis=0,  # 纵向拼接
        ignore_index=True  # 重置索引
    )

    cei_df.columns = cei_df.columns.str.strip()
    cei_df['substation_code']=cei_df['substation_code'].str.strip()
        # 数据缺失处理，后续重读数据
    cei_df = cei_df[cei_df['period_time'] != 220000]
    # 定义period_time处理函数
    def format_period_time(x):
        if x == 0:
            return x
        else:
            return x/10000

    # 应用函数到列
    cei_df['period_time'] = cei_df['period_time'].apply(format_period_time)
    cei_df['timestamp'] = pd.to_datetime(cei_df['period_date']) + pd.to_timedelta(cei_df['period_time'], unit='h')

    # dates = sorted(cei_df['period_date'].unique())
    ########## 线路有功 ####################
    ## branch p and q
    branch_path = os.path.join(data_dir, f"线路有功.xlsx")
    branch_df = pd.read_excel(branch_path, sheet_name=None)
    # # 获取第一个 sheet
    # sheet = branch_df[list(branch_df.keys())[0]]
    # sheet.columns = sheet.columns.str.strip()  # 去除列名空格
    # sheet['ptnum_type'] = sheet['ptnum_type'].str.strip()
    # 合并所有 sheet 的数据
    all_sheets = []
    for sheet in branch_df.values():
        sheet.columns = sheet.columns.str.strip()  # 去除列名空格
        for col in sheet.columns:
            if isinstance(sheet[col].iloc[0], str):
                sheet[col] = sheet[col].str.strip()
        all_sheets.append(sheet)
    sheet = pd.concat(all_sheets, ignore_index=True)

    ########### 档案###############
    edge_path = os.path.join(data_dir, f"档案.xlsx")
    edge_df = pd.read_excel(edge_path, sheet_name=None)

    ## 场站
    changzhan_column_keys = edge_df['场站'].columns.tolist()
    node_index = edge_df['场站'][changzhan_column_keys[0]].str.strip() # 场站编号
    node_type = edge_df['场站'][changzhan_column_keys[1]].str.strip().unique()  # 场站类型
    mapping = {text:i for i, text in enumerate(node_type)}  # 场站类型映射
    node_type_encoder = edge_df['场站'][changzhan_column_keys[1]].str.strip().map(mapping)  # 场站类型编码
    num_node = len(node_index)

    ## 线路
    xianlu_column_keys = edge_df['线路'].columns.tolist()
    edge_index_all = edge_df['线路'][xianlu_column_keys[0]].str.strip().tolist()  # 线路编码，第一列
    start_node = edge_df['线路'][xianlu_column_keys[1]].str.strip().tolist()  # 起始节点
    end_node = edge_df['线路'][xianlu_column_keys[3]].str.strip().tolist()  # 终止节点
    ## 电源
    gen_column_keys = edge_df['电源'].columns.tolist()
    gen_index = edge_df['电源'][gen_column_keys[0]].tolist()
    num_gen = len(gen_index)

    ####################### 判断线路起止点是否在node_index中 #######################
    node_set = set(node_index.tolist())
    gen_set = set(gen_index)

    # 初始化计数
    start_in_node = sum(1 for x in start_node if x in node_set)
    start_in_gen = sum(1 for x in start_node if x in gen_set)
    start_not_in_any = sum(1 for x in start_node if x not in node_set and x not in gen_set)

    end_in_node = sum(1 for x in end_node if x in node_set)
    end_in_gen = sum(1 for x in end_node if x in gen_set)
    end_not_in_any = sum(1 for x in end_node if x not in node_set and x not in gen_set)

    # 构建 DataFrame 表格
    df = pd.DataFrame({
        'in node_index': [start_in_node, end_in_node],
        'in gen_index': [start_in_gen, end_in_gen],
        'not in any': [start_not_in_any, end_not_in_any]
    }, index=['start_node', 'end_node'])

    print(df.head())
###################################
################ 特征组织 ######################
###################################

    ## 提取 edge_feature_matrix 和 edge_index
    # 提取测量类型为 P、Q、I，side_type 为 4 的数据
    filtered_df = sheet[(sheet['side_type'] == 4) & (sheet['ptnum_type'].isin(['P', 'Q']))].copy()
    # 有效时间列 v1 ~ vN（比如 v1~v24）
    time_columns = [col for col in sheet.columns if col.startswith('v') and col[1:].isdigit()]
    # 清洗：空白/NULL → np.nan
    filtered_df[time_columns] = filtered_df[time_columns].replace(
        to_replace=[r'^\s*$', r'(?i)null'], value=np.nan, regex=True
    ).astype(float)
    print(np.sum(np.sum(filtered_df[time_columns]==np.nan)))
    # 所有线路编号和日期
    line_codes = filtered_df['code'].str.strip().unique()
    dates = sorted(filtered_df['period_date'].unique())
    line_code_to_index = {code: i for i, code in enumerate(line_codes)}
    num_lines = len(line_codes)
    num_days = len(dates)
    num_time_steps = 23
    num_features = 2  # P/Q
    values_used = set(edge_index_all)  # 所有线路编码的值


    T = num_days * num_time_steps # 总时间步数
    E = num_lines # 总线路数
    F = num_features # 特征数（P、Q、I）

    #### 初始化特征矩阵和掩码矩阵 edge_feature_matrix
    edge_feature_matrix = np.zeros((T, E, F), dtype=np.float32)
    edge_mask_matrix = np.zeros((T, E, F), dtype=np.float32)
    # 遍历组织
    for d_idx, date in enumerate(dates):
        for code in line_codes:
            edge_idx = line_code_to_index[code] # 获取线路编码对应的索引
            for f_idx, ptype in enumerate(['P', 'Q']):
                row = filtered_df[
                    (filtered_df['code'] == code) &
                    (filtered_df['ptnum_type'] == ptype) &
                    (filtered_df['period_date'] == date)
                    ]
                if not row.empty:
                    values = row[time_columns].values[0]
                    masks = ~np.isnan(values)
                    for h in range(num_time_steps):
                        t = d_idx * num_time_steps + h
                        edge_feature_matrix[t, edge_idx, f_idx] = values[h] if masks[h] else 0.0
                        edge_mask_matrix[t, edge_idx, f_idx] = 1.0 if masks[h] else 0.0
    edge_feature_matrix = torch.tensor(edge_feature_matrix,dtype=torch.float32)
    #### 初始化节点特征矩阵 node_feature_matrix

    ######################## 构造节点特征 ########################

    # 1. edge_index 构造（2 × E tensor）
    # 过滤掉不在 edge_index_all 中的线路编码
    edge_code_filtered = [code for code in line_code_to_index.keys() if code in edge_index_all]
    edge_id_map = {code: i for i, code in enumerate(edge_code_filtered)}

    #
    start_node_filtered = [node for node in start_node if node in node_index.tolist()]
    end_node_filtered = [node for node in end_node if node in node_index.tolist()]
    node_list = sorted(set(start_node_filtered) | set(end_node_filtered))  # 所有节点
    node_id_map = {node: idx for idx, node in enumerate(node_list)}  # 编号映射

    #构造节点索引
    edge_list_filtered = [
        (node_id_map[start_node_filtered[i]], node_id_map[end_node_filtered[i]])
        for i, c in enumerate(edge_code_filtered)
    ]

    edge_index_filtered = torch.tensor(edge_list_filtered, dtype=torch.long).T  # 转置为 [2, E] 形式

    # 3. 边特征（如 P/Q/I 时间序列） → 维度 E × F
    edge_attr = torch.tensor(edge_feature_matrix, dtype=torch.float)  # 示例 shape: [E, 24]

    # 4. 节点特征（如聚合的边信息、节点静态属性等）
    node_feature_matrix = build_node_feature_matrix(
        edge_feature_matrix, edge_index_filtered.numpy(), node_type_encoder.to_numpy()
    )  # shape: [T, N, 6]
    node_features = torch.tensor(node_feature_matrix, dtype=torch.float)  # shape: [N, F_node]
    # print(edge_index_filtered)

    # 5. 节点标签（碳因子） y，维度 N × 1
    # node_labels = torch.tensor([carbon_map[node] for node in node_list], dtype=torch.float).view(-1, 1)
    T = node_feature_matrix.shape[0]
    N = node_feature_matrix.shape[1]

    node_labels = torch.zeros((T, N, 1), dtype=torch.float32)  # [T, 252, 1]

    # 记录每个时间戳对应的索引
    time_index_map = dict()
    for i, time_time in enumerate(sorted(cei_df['timestamp'].unique())):
        time_index_map[time_time] = i

    grouped_by_time = cei_df.groupby('timestamp')

    for time_time, group in grouped_by_time:
        time_index = time_index_map[time_time]
        for _, row in group.iterrows():
            substation_code = row['substation_code']
            if substation_code in node_id_map:
                node_index = node_id_map[substation_code]
                cei_value = row['cei']
                log_cei = np.log1p(cei_value + 1.01)
                nan_mask = np.isnan(log_cei)
                if nan_mask.any():
                    print(f"index{row}, original cei value {cei_value}")
                node_labels[time_index, node_index, 0] = log_cei

    print(node_labels[:3,:3])

    # train and test
    train_bus = node_feature_matrix[:int(num_cases * train_test_rate)]
    test_bus = node_feature_matrix[int(num_cases * train_test_rate):]
    train_branch = edge_feature_matrix[:int(num_cases * train_test_rate)]
    test_branch = edge_feature_matrix[int(num_cases * train_test_rate):]
    train_labels = node_labels[:int(num_cases * train_test_rate)]
    test_labels = node_labels[int(num_cases * train_test_rate):]

    # normalize
    bool_columns = [2,3,4]
    train_bus, bus_min, bus_max = minmax_scale_with_bool(train_bus, bool_columns, eps=1e-8)
    train_branch, branch_min, branch_max = minmax_scale_with_bool(train_branch, [], eps=1e-8)
    train_labels, labels_min, labels_max = minmax_scale_with_bool(train_labels, [], eps=1e-8)
    ####训练集增强
    train_transform = Compose([SmallNoiseEdgeAttr(), MaskNodeFeature(mask_ratio=0.05)])

    ############### 构造 TemporalData 对象 ###############
    ##########################
    # 3. 构造 TemporalData
    ##########################
    base_edge_index = edge_index_filtered.T.clone()

    data_list_train = []
    n_train = node_feature_matrix.shape[0]
    for t in range(n_train):
        data_list_train.append(
            TemporalData(
                x=train_bus[t], edge_index=(base_edge_index.clone()).long(),
                edge_attr=train_branch[t].clone(), y=train_labels[t]
            )
        )
    train_dataset = ListDataset(data_list_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(data_list_test, batch_size=BATCH_SIZE)

    ############### 训练GNN#################
    model = CarbonGNN(in_nf=node_feature_matrix.size(-1), edge_nf=edge_feature_matrix.shape[2])
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.5) #
    scaler = GradScaler('cuda')
    loss_fn = nn.HuberLoss(delta=1.0)
    #
    # ---------------- training loop ------------------
    best_loss, patience = float('inf'), 0
    train_loss_list = []
    for epoch in range(EPOCHS):
        # ---- train ----
        model.train()
        running = 0
        for batch in train_loader:
            batch = batch.to(device)
            # print(batch.edge_index.shape)  # ➜ torch.Size([2, 1472])
            # print(batch.edge_attr.shape)  # ➜ torch.Size([1472, 2])
            optim.zero_grad(set_to_none=True)
            with autocast():
                pred = model(batch.x, batch.edge_index.T, batch.edge_attr)
                loss = loss_fn(pred, batch.y.squeeze(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(optim)
            scaler.update()
            running += loss.item() * batch.y.size(0)
        train_loss_list.append(running)
        sched.step()
        if epoch % 20 == 0:
            print(f"epoch {epoch:03d} | loss={loss / len(data_list_train):.6f}")
        train_loss = running / len(train_loader.dataset)

        # ---- val ----
        model.eval()
        running = 0
        with torch.no_grad(), autocast():
            for batch in train_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index.T, batch.edge_attr)
                running += loss_fn(pred, batch.y).item() * batch.y.size(0)
        val_loss = running / len(train_loader.dataset)
        sched.step()
        print(f"Ep {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss + 1e-4 < best_loss:
            best_loss, patience = val_loss, 0
            torch.save(model.state_dict(), "best_gnn.pt")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.");
                break

    # ---------------- evaluation ---------------------
    model.load_state_dict(torch.load("best_gnn.pt"))
    model.eval()
    preds_list = []
    with torch.no_grad(), autocast():
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index.T, batch.edge_attr)
            preds_list.append(pred.cpu())

    # define inverse normalization using lambda function

    ## plot for train
    preds = torch.cat(preds_list, dim=0)
    true_list = [batch.y.cpu() for batch in train_loader]
    true = torch.cat(true_list, dim=0)
    fig, ax = plt.subplots()
    plt.hist(true,bins = 50)

    plt.figure(figsize=(10, 4))
    plt.plot(true.numpy()[:100], label="True", linestyle="--", marker=".")
    plt.plot(preds.numpy()[:100], label="Pred", alpha=0.7)
    plt.legend()
    title_str = f"Epochs:{EPOCHS} train_num:{n_train} batch_size:{BATCH_SIZE}"
    plt.title(title_str)
    plt.tight_layout()
    plt.show()

    print(f"Total time: {time.time() - start_time:.1f}s | Best val loss (std‑scaled) {best_loss:.4f}")

    plt.plot(train_loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Loss over Epochs')
    plt.show()
# print("x:", td.x.shape, td.x.dtype)
# print("edge_index:", td.edge_index.shape, td.edge_index.dtype)
# print("edge_attr:", td.edge_attr.shape, td.edge_attr.dtype)
# print("y:", td.y.shape, td.y.dtype)