import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import BaseTransform
import random

def minmax_scale_with_bool(x, bool_columns, type='minmax', eps=1e-8):
    """
    对连续特征进行 MinMax 归一化，保留布尔特征原样
    :param x: Tensor, shape (T, N, n)
    :param bool_columns: 布尔特征所在的列索引（例如 [2, 5] 表示第 2、5 列是布尔值）
    :param eps: 防止除零的平滑常数
    :return: 归一化后的张量
    """
    # 先保存布尔特征
    bool_data = x[:, :, bool_columns]

    # 归一化连续特征，排除布尔列
    continuous_data = x.clone()
    continuous_data[:, :, bool_columns] = 0  # 将布尔列设为 0（不参与归一化）
    if type == 'minmax':
        # 对连续特征进行 MinMax 归一化
        x_min = continuous_data.amin(dim=(0, 1), keepdim=True)
        x_max = continuous_data.amax(dim=(0, 1), keepdim=True)

        # 归一化
        continuous_data = (continuous_data - x_min) / (x_max - x_min + eps)
    elif type == 'log':
        continuous_data = torch.log(continuous_data)
        x_min = 1
        x_max = 2
    # 恢复布尔特征
    x_normalized = continuous_data
    x_normalized[:, :, bool_columns] = bool_data
    return x_normalized, x_min, x_max


def minmax_inverse_with_bool(x_scaled, x_min, x_max, bool_columns, eps=1e-8):
    """
    反归一化：恢复 MinMax 归一化后的数据到原始范围
    :param x_scaled: 归一化后的数据，Tensor, shape (T, N, n)
    :param x_min: 最小值，Tensor, shape (1, 1, n)
    :param x_max: 最大值，Tensor, shape (1, 1, n)
    :param bool_columns: 布尔特征的列索引，List or Tensor，形如 [2, 5]
    :param eps: 防止除零的平滑常数
    :return: 恢复后的数据
    """
    # 先恢复布尔特征列
    bool_data = x_scaled[:, :, bool_columns]

    # 恢复连续特征的部分
    continuous_data = x_scaled.clone()

    # 反归一化连续特征
    continuous_data = continuous_data * (x_max - x_min + eps) + x_min

    # 恢复布尔特征列
    continuous_data[:, :, bool_columns] = bool_data

    return continuous_data


def build_node_feature_matrix(edge_feature_matrix, edge_index, node_types):
    """
    构造节点特征矩阵，用于图神经网络建模。

    参数：
    - edge_feature_matrix: np.ndarray [T, E, 3]，每条边的 [P, Q, I]
    - edge_index: np.ndarray [2, E]，每条边的起点与终点节点编号
    - node_types: np.ndarray [N]，每个节点的类型编号（0=负荷，1=发电，2=中继）

    返回：
    - node_feature_matrix: np.ndarray [T, N, 6]，每个节点每个时刻的六维特征
    """
    T, E, F = edge_feature_matrix.shape
    N = len(node_types)
    node_feature_matrix = np.zeros((T, N, 5), dtype=np.float32)
    for t in range(T):
        P_in = np.zeros(N)
        # I_in = np.zeros(N)
        degree = np.zeros(N)

        for e in range(E):
            src = edge_index[0, e]
            tgt = edge_index[1, e]

            P = edge_feature_matrix[t, e, 0]  # 有功
            # I = edge_feature_matrix[t, e, 2]  # 电流

            P_in[tgt] += P
            # I_in[tgt] += I

            degree[src] += 1
            degree[tgt] += 1

        # 构造 one-hot 类型
        type_onehot = np.zeros((N, 3), dtype=np.float32)
        for i in range(N):
            if 0 <= node_types[i] < 3:
                type_onehot[i, node_types[i]] = 1.0

        # 拼接特征
        node_feature_matrix[t, :, 0] = P_in
        # node_feature_matrix[t, :, 1] = I_in
        node_feature_matrix[t, :, 1] = degree
        node_feature_matrix[t, :, 2:] = type_onehot

    return torch.tensor(node_feature_matrix, dtype=torch.float32)

###
# • 随机遮蔽 5 % 节点特征
# • 对边权加 ±1 % 噪声
# • 随机时间平移/剪裁序列
# 可“制造”更多工况。
class SmallNoiseEdgeAttr(BaseTransform):
    """对边特征加 ±1% 噪声"""
    def __call__(self, data: Data):
        if self.training:                       # 仅训练集启用
            noise = (torch.rand_like(data.edge_attr) - 0.5) * 0.02
            data.edge_attr = data.edge_attr * (1 + noise)
        return data


class MaskNodeFeature(BaseTransform):
    """随机遮掉 5% 节点某些特征"""
    def __init__(self, mask_ratio=0.05):
        self.mask_ratio = mask_ratio

    def __call__(self, data: Data):
        if self.training:
            N, F = data.x.size()
            num_mask = int(N * F * self.mask_ratio)
            idx = torch.randint(0, N * F, (num_mask,))
            flat = data.x.flatten()
            flat[idx] = 0.0
            data.x = flat.view(N, F)
        return data


class ListDataset(Dataset):
    def __init__(self, data_list, transform=None):
        super().__init__(transform=transform)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]
        # transform 会在这里自动触发
        return data