import torch
import pandas as pd
import numpy as np
# 支路潮流矩阵 PB(t)  &  PN(t)  (忽略网损用直流潮流) -----------------
path = "D://OneDrive - mails.tsinghua.edu.cn//CarbonFactor//results//"
class EN():
    def __init__(self, N, K, key2branch, OFFSET, mapping, EG,
                 branch_dict=None, eps=1e-6):
        super(EN, self).__init__()
        self.N = N
        self.K = K
        self.key2branch = key2branch
        self.OFFSET = OFFSET
        self.mapping = mapping
        self.EG = EG
        self.eps = 1e-6
        self.branch_dict = branch_dict
        self.EG = EG

    def build_EN(self, edge_attr_t, PGt_t, eps=1e-6):
        """可逆保障: 若 diag==0 → 加 ε ；不可逆仍报错则改用 pseudo-inverse"""
        # ---------- PB(t) ----------
        PB = torch.zeros((self.N, self.N))
        for idx, (u, v) in enumerate(self.key2branch.keys()):  # edge_index.T
            P = edge_attr_t[idx, 0]  # 取 Pij
            if P>0:
                PB[u - self.OFFSET, v - self.OFFSET] = P
            else:
                PB[v - self.OFFSET, u - self.OFFSET] = -P
        PB_array = PB.numpy()
        # PB_df = pd.DataFrame(PB_array, columns=[f"c{i}" for i in range(PB_array.shape[1])])
        # PB_df.to_csv(path+"PB.csv", index=False)  # 保存支路潮流矩阵
        # ---------- PG(t) ----------
        PG = torch.zeros((self.K,self.N), dtype=torch.float)  # [K, N]
        PG[torch.arange(self.K), self.mapping["Bus"]] = PGt_t
        # PG_array = PG.detach().cpu().numpy()
        # PG_df = pd.DataFrame(PG_array, columns=[f"c{i}" for i in range(PG_array.shape[1])])
        # PG_df.to_csv(path + "PG.csv", index=False)
        # ---------- PN(t) ----------
        PN_diag = PB.sum(0) + PG.sum(0)  # 仅入流 + 发电注入
        PN_diag = torch.where(PN_diag == 0, PN_diag + eps, PN_diag)  # 对角补 ε
        PN = torch.diag(PN_diag)
        # PN_array = PN.detach().cpu().numpy()
        # PN_df = pd.DataFrame(PN_array, columns=[f"c{i}" for i in range(PN_array.shape[1])])
        # PN_df.to_csv(path + "PN.csv", index=False)
        A = PN - PB.T  # 理论中的可逆矩阵
        try:
            # print("A:", A.dtype)
            # print("PG:", PG.dtype)
            # print("EG:", self.EG.dtype)
            EN = torch.linalg.solve(A, PG.T @ self.EG)  # 比 inv @ b 数值更稳
        except torch._C._LinAlgError:
            # 仍奇异：退化到 Moore–Penrose 伪逆
            EN = torch.linalg.pinv(A) @ (PG.T @ self.EG)
        # EN_array = EN.detach().cpu().numpy()
        # EN_df = pd.DataFrame(EN_array, columns=[f"c{i}" for i in range(1)])
        # EN_df.to_csv(path + "EN.csv", index=False)
        return EN
