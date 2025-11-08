from torch_geometric.nn import GATv2Conv
import torch.nn as nn
import torch.nn.functional as Func

HID = 128 #
head_num = 8
DROP_P = 0.2
################## 定义 GNN 模型 ##################
class CarbonGNN(nn.Module):
    def __init__(self, in_nf, edge_nf, hid=HID, out_nf=1):
        super().__init__()
        self.fc_in = nn.Linear(in_nf, hid) if in_nf != hid else nn.Identity()
        self.conv1 = GATv2Conv(hid, hid//head_num, edge_dim=edge_nf, heads=head_num, concat=True, dropout=DROP_P)
        self.norm1 = nn.LayerNorm(hid)
        # self.conv2 = GATv2Conv(hid, hid, edge_dim=edge_nf, heads=8, concat=True, dropout=DROP_P)
        # self.norm2 = nn.LayerNorm(hid)
        self.drop = nn.Dropout(DROP_P)
        self.out = nn.Sequential(nn.Linear(hid, hid),nn.ReLU(),nn.Linear(hid, out_nf)
)

    def forward(self, x, edge_index, edge_attr):
        x = self.fc_in(x)
        h = Func.leaky_relu(self.conv1(x, edge_index, edge_attr))
        h = self.norm1(h + x)  # residual + LN
        # h2 = Func.leaky_relu(self.conv2(h, edge_index, edge_attr))
        # h = self.norm2(h2 + h)
        h = self.drop(h)
        return Func.softplus(self.out(h)).squeeze(-1) # 防止非负值出现

# from torch_geometric.nn import GCNConv
# import torch.nn as nn
# import torch.nn.functional as Func

# HID = 128
# DROP_P = 0.2

#
# ################## 定义 GCN 模型 ##################
# class CarbonGNN(nn.Module):
#     def __init__(self, in_nf, edge_nf, hid=HID, out_nf=1):
#         super().__init__()
#         self.fc_in = nn.Linear(in_nf, hid) if in_nf != hid else nn.Identity()
#
#         # GCNConv 不使用 edge_attr，接口只接受 x 和 edge_index
#         self.conv1 = GCNConv(hid, hid)
#         self.norm1 = nn.LayerNorm(hid)
#
#         # 你可以继续加层，这里保留单层结构
#         self.drop = nn.Dropout(DROP_P)
#         self.out = nn.Sequential(
#             nn.Linear(hid, hid),
#             nn.ReLU(),
#             nn.Linear(hid, out_nf)
#         )
#
#     def forward(self, x, edge_index, edge_attr=None):  # edge_attr 不再使用
#         x = self.fc_in(x)
#         h = Func.relu(self.conv1(x, edge_index))
#         h = self.norm1(h + x)  # 残差连接 + LN
#         h = self.drop(h)
#         return Func.softplus(self.out(h)).squeeze(-1)
