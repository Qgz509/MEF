"""
build_topology.py

从暂态潮流 Excel 表中解析 Pij/Pji 列名，
抽取“支路编号 – 起止节点”信息，
生成 (1) 支路列表 DataFrame  (2) NetworkX 图并可视化。

使用方法:
    python build_topology.py  input.xlsx  [--plot]  [--outfile topo.png]

依赖:
    pip install pandas networkx matplotlib openpyxl
"""

import re
from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def parse_branches(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 DataFrame 列名中提取支路与节点信息。

    列名示例:
        Pij(AC17-BUS-26)   表示支路 AC17 从某节点 -> BUS 26
        Pji(AC17-BUS-16)   表示支路 AC17 从 BUS 26 -> BUS 16
    """
    pij_cols = [c for c in df.columns if isinstance(c, str) and c.startswith(("Pij", "Pji"))]
    branch_dict = {}

    pattern = re.compile(r"P[ij][^)]*\(([A-Z0-9\+]+)-BUS-(\d+)\)")
    for col in pij_cols:
        m = pattern.match(col)
        if m:
            line_id, bus = m.group(1), int(m.group(2))
            direction = "from" if col.startswith("Pij") else "to"
            branch_dict.setdefault(line_id, {})[direction] = bus

    # 只保留同时拥有 from 与 to 的完整记录
    edges = [
        {"Bus A": d["from"], "Bus B": d["to"], "Branch ID": lid}
        for lid, d in branch_dict.items()
        if {"from", "to"} <= d.keys()
    ]
    return pd.DataFrame(edges).sort_values("Branch ID")


def build_graph(edges_df: pd.DataFrame) -> nx.Graph:
    """根据 DataFrame 构建无向图并写入支路编号 label。"""
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row["Bus A"], row["Bus B"], label=row["Branch ID"])
    return G


def plot_topology(G: nx.Graph, outfile: Path):
    """绘制拓扑，可保存为文件。"""
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=300)
    nx.draw_networkx_edges(G, pos, width=1.2)
    nx.draw_networkx_labels(G, pos, font_size=8)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.axis("off")
    plt.title("Power System Topology (Buses and Branch IDs)")
    if outfile:
        plt.savefig(outfile, bbox_inches="tight", dpi=300)
        print(f"Topology figure saved to {outfile.resolve()}")
    else:
        plt.show()
    plt.close()

