import numpy as np
import sys
import uuid
import time
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import csv
import io


node_db = pd.read_excel("D:\OneDrive - mails.tsinghua.edu.cn//CarbonFactor//宋博士_碳流输入输出数据//入模表//node_v1.xlsx",header=0)
node_db_list = []
node_db_num = 0

for record in range(len(node_db)):
    node_db_list.insert(node_db_num,node_db.loc[record])
    node_db_num = node_db_num+1

print("node_db_num = " + str(node_db_num))

line_db = pd.read_excel("D:\OneDrive - mails.tsinghua.edu.cn//CarbonFactor//宋博士_碳流输入输出数据//入模表//line_v1.xlsx",header=0)
line_db_list = []
line_db_num = 0

for record in range(len(line_db)):
    line_db_list.insert(line_db_num,line_db.loc[record])
    line_db_num = line_db_num+1

"""节点数据处理"""
# 2、数据库数据处理解析
node_data_list = []
# 节点数据取出相应的列
number = 1
for index in node_db_list:
    temp = [index[0], number, index[3], index[7], index[4], index[9], float(index[8]), 0]
    node_data_list.append(temp)
    # node_id='节点ID', '统计number',node_type='厂站类型', node_pg='厂节点有功发电', node_cei='节点碳排放因子', area_num='对区域编号的引用', node_p='站节点有功', 0
    # node_data_list = 0:node_id 1:node_num 2:node_type 3:node_genP 4:node_cei 5:area_id 6:node_P 7:import_flag
    number = number + 1

# 节点数据取出正在线路中使用的节点
# node_cal_list为剔除空节点后的节点数据列表（也就是节点在线段的起点或者终点表示该节点不是空结点）
node_cal_list = []
for node_data_item in node_data_list:
    # line_id='线路ID';data_date='数据日期';time='数据时刻';line_p='线路有功量测';start_nid='线路起始厂站ID';end_nid='线路终止厂站ID';line_name='线路中文全称';
    for index in line_db_list:
        # start_nid = '线路起始厂站ID' == node_id = '节点ID'
        if index[4] == node_data_item[0]:
            if node_data_item not in node_cal_list:
                node_cal_list.append(node_data_item)
        if index[5] == node_data_item[0]:
            if node_data_item not in node_cal_list:
                node_cal_list.append(node_data_item)

# 节点数据新编号
number = 1
for index in node_cal_list:
    index[1] = number
    number = number + 1
    # node_id='节点ID', '统计number',node_type='厂站类型', node_pg='厂节点有功发电', node_cei='节点碳排放因子', area_num='对区域编号的引用', node_p='站节点有功', 0
    # node_cal_list = 0:node_id 1:node_num 2:node_type 3:node_genP 4:node_cei 5:area_id 6:node_P 7:import_flag



"""线路数据处理"""
# '线路ID'; '线路自增num'; '起始场站ID'; '终止场站ID'; '线路有功量测'.
# line_data_list = 0:line_id 1:line_num 2:start_num 3:end_num 4:line_P
line_data_list = []
number = 1
# line_id='线路ID';data_date='数据日期';time='数据时刻';line_p='线路有功量测';start_nid='线路起始厂站ID';end_nid='线路终止厂站ID';line_name='线路中文全称';
for index in line_db_list:
    start_num = 0
    end_num = 0
    # node_id='节点ID', '统计number',node_type='厂站类型', node_pg='厂节点有功发电', node_cei='节点碳排放因子', area_num='对区域编号的引用', node_p='站节点有功', 0
    for node_data_item in node_cal_list:
        if index[4] == node_data_item[0]:
            start_num = node_data_item[1]
        if index[5] == node_data_item[0]:
            end_num = node_data_item[1]
    temp = [index[0], number, start_num, end_num, index[3]]
    # temp = [index[0], number, int(index[4]), int(index[5]), index[3]]
    line_data_list.append(temp)
    # line_data_list = 0:line_id 1:line_num 2:start_num 3:end_num 4:line_P
    number = number + 1
print(line_data_list)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                       矩阵生成部分     
# line_data_list = 0:line_id 1:line_num 2:start_num 3:end_num 4:line_P
# node_cal_list = 0:node_id 1:node_num 2:node_type 3:node_genP 4:node_cei  5:area_num                                                                                                     
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""从xlsx中读取数据"""
#node_cal_list = pd.read_excel('node.xlsx', sheet_name='Sheet1')
#line_data_list = pd.read_excel('line.xlsx', sheet_name='Sheet1')
# area_data_list = pd.read_excel('demo_data/area.xlsx', sheet_name='Sheet1')

# print(node_cal_list.info())

# node_cal_list = node_cal_list.values.tolist()
# line_data_list = line_data_list.values.tolist()
# # print(line_data_list)
# area_data_list = area_data_list.values.tolist()


M = len(node_cal_list)  # 节点数量
N = len(line_data_list)  # 线路数量

print("节点数量：", M)
print("线路数量：", N)

# 矩阵初始化
P_G = np.zeros((M, M), dtype=float)
P_Z = np.zeros((M, M), dtype=float)
E_G = np.zeros((M, 1), dtype=float)
P_L = np.zeros((M, M), dtype=float)
P_B = np.zeros((M, M), dtype=float)

"""
P_L 是对角矩阵

1，矩阵横纵坐标表示节点node_num-1
2，变电站类型的节点：
    -如果节点是某条线路的终止节点，为流入。P_L[node_num-1][node_num-1]记录的是流入量（正值）。厂节点有功发电node_genP为0，node_cei（碳排因子）为0。
    -如果节点是某条线路的起始节点，为流出。P_L[node_num-1][node_num-1]记录的站节点有功node_P（abs值）。
        node_genP=流出电量 + 站节点有功node_P，node_cei（碳排因子）为0.5839。import_flag 标记为1。

3，其他场站类型节点：
    -如果节点为流出。P_L[node_num][node_num] = 站节点有功node_P。node_genP = 站节点有功node_P - 外部供电量流出（负值）

    -如果节点为流入。如果流入小于 站节点有功node_P，也就是node_genP>0,P_L[node_num-1][node_num-1]记录的是站节点有功node_P。
                  如果流入大于 站节点有功node_P，也就是node_genP<0,P_L[node_num-1][node_num-1]记录的是流入量（正值）。node_genP设为0。
                        node_genP = 站节点有功node_P(abs) - 外部供电量流入（正值）
"""
# 矩阵赋值
P_import = 0
for node_data_item in node_cal_list:
    print(node_data_item[3])
    print(",")
    tempdata = 0
    if int(node_data_item[2]) == 3:  # 如果是变电站类型
        #node_data_item[3] = 0  # 有功发电量为0
        #node_data_item[4] = 0  # 碳排放因子为0
        for line_data_item in line_data_list:
            if node_data_item[1] == line_data_item[2]:  # 如果节点是起始节点
                # 如果是某条线路的起始节点，为流出
                # P_L[node_num][node_num] = P_L[node_num][node_num] - 线路有功量测
                P_L[int(node_data_item[1] - 1), int(node_data_item[1] - 1)] = P_L[int(node_data_item[1] - 1), int(
                    node_data_item[1] - 1)] - float(line_data_item[4])
            elif node_data_item[1] == line_data_item[3]:
                # 如果是某条线路的终止节点，为流入
                # P_L[node_num][node_num] = P_L[node_num][node_num] + 线路有功量测
                P_L[int(node_data_item[1] - 1), int(node_data_item[1] - 1)] = P_L[int(node_data_item[1] - 1), int(
                    node_data_item[1] - 1)] + float(line_data_item[4])
        # supply by import electricity 如果节点从外部供电
        if P_L[int(node_data_item[1] - 1), int(node_data_item[1] - 1)] < 0:
            #node_data_item[4] = 0.5839  # 碳排放因子赋值0.5839
            # node_genP = 外部供电量 + 站节点有功node_P
           # node_data_item[3] = abs(P_L[int(node_data_item[1] - 1), int(node_data_item[1] - 1)]) + abs(
              # float(node_data_item[6]))
            # P_L[node_num][node_num] = 站节点有功node_P
            P_L[int(node_data_item[1] - 1), int(node_data_item[1] - 1)] = abs(float(node_data_item[6]))
            # p_import = pimport + (node_genP = 外部供电量 + 站节点有功node_P)
            P_import = P_import + node_data_item[3]
            # import_flag 标记为1 ，流出大于流入，从外部供电的节点。
            node_data_item[7] = 1

    else:  # if it is not a electricity substation如果是非变电站类型
        for line_data_item in line_data_list:
            # 如果是某条线路的起始节点，为流出
            if node_data_item[1] == line_data_item[2]:
                P_L[int(node_data_item[1] - 1), int(node_data_item[1] - 1)] = P_L[int(node_data_item[1] - 1), int(
                    node_data_item[1] - 1)] - float(line_data_item[4])
            # 如果是某条线路的终止节点，为流入
            elif node_data_item[1] == line_data_item[3]:
                P_L[int(node_data_item[1] - 1), int(node_data_item[1] - 1)] = P_L[int(node_data_item[1] - 1), int(
                    node_data_item[1] - 1)] + float(line_data_item[4])

        if P_L[int(node_data_item[1] - 1), int(node_data_item[1] - 1)] < 0:  # support the network  流出
            # node_genP = 站节点有功node_P - 外部供电量流出（负值）
            #node_data_item[3] = abs(float(node_data_item[6])) - P_L[
              #  int(node_data_item[1] - 1), int(node_data_item[1] - 1)]
            # P_L[node_num][node_num] = 站节点有功node_P
            P_L[int(node_data_item[1] - 1), int(node_data_item[1] - 1)] = abs(float(node_data_item[6]))
        else:  # supply by the network 流入
            # node_genP = 站节点有功node_P - 外部供电量流入（正值）
          #  node_data_item[3] = abs(float(node_data_item[6])) - P_L[
           #     int(node_data_item[1] - 1), int(node_data_item[1] - 1)]
            if node_data_item[3] < 0:  # 流出大于站节点有功node_P，则节点有功node_genP为0
                node_data_item[3] = 0
            else:  # P_L[node_num][node_num] = 站节点有功node_P
                P_L[int(node_data_item[1] - 1), int(node_data_item[1] - 1)] = abs(float(node_data_item[6]))
print(node_data_item[3])

"""
P_B 是稀疏矩阵，储存节点之间的线路有功量测值
1，矩阵的横坐标表示线路的起始节点num-1,纵坐标表示线路的终止节点num-1。
2，矩阵的值表示线路的有功量测值（abs）。
"""
for line_data_item in line_data_list:
    # print(line_data_item,"===================")
    n_p = float(line_data_item[4])
    if n_p >= 0:
        P_B[int(line_data_item[2] - 1), int(line_data_item[3] - 1)] = n_p + P_B[
            int(line_data_item[2] - 1), int(line_data_item[3] - 1)]
    else:
        P_B[int(line_data_item[3] - 1), int(line_data_item[2] - 1)] = P_B[int(line_data_item[3] - 1), int(
            line_data_item[2] - 1)] - n_p

"""
P_G 是对角矩阵，储存节点的 厂节点有功发电node_genP
1，矩阵的横纵坐标表示节点num-1。
2，矩阵的值表示节点的 厂节点有功发电node_genP。如果node_genP>0记录进去，否则设置为0。

"""
for node_data_item in node_cal_list:
    print(node_data_item[3])
    if node_data_item[3] >= 0:
        P_G[int(node_data_item[1] - 1), int(node_data_item[1] - 1)] = node_data_item[3]
    else:
        node_data_item[3] = 0
print(P_G)
"""
P_Z 是高矩阵[2x,1x]型，上半部分是P_B，下半部分是P_G
1,P_B是稀疏矩阵，储存节点之间的线路有功量测值
2,P_G是对角矩阵，储存节点的 厂节点有功发电node_genP
"""
P_Z = P_B+P_G

"""
E_G 是单列矩阵，储存节点的 node_cei 碳排因子
"""
for node_data_item in node_cal_list:
    E_G[int(node_data_item[1] - 1), 0] = node_data_item[4]

"""
P_N 是对角矩阵
1，2*M 的单位长矩阵 点乘 P_Z 高矩阵
2，P_Z 高矩阵 是上半部分是P_B是稀疏矩阵，储存节点之间的线路有功量测值。P_G是对角矩阵，储存节点的 厂节点有功发电node_genP。
3，点乘的效果是 P_Z 同一列中的数据相加成一个数。物理意义是 节点的厂节点有功发电node_genP + 以节点为起点的所有线路有功量测值。
4，形成[1,M]的矩阵，每个点表示节点的 厂节点有功发电node_genP + 以节点为起点的所有线路有功量测值。
5，在转换为对角矩阵，横纵坐标表示节点num-1，值表示节点的 厂节点有功发电node_genP + 以节点为起点的所有线路有功量测值。
"""
P_N = np.diag(np.dot(np.ones((1,M), dtype=np.float64), P_Z)[0, :]+0.00001)

# print(P_N)
nownow = datetime.datetime.now()
print(nownow)
print("matrix production")
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                       结果运算部分 
# line_data_list = 0:line_id 1:line_num 2:start_num 3:end_num 4:line_P
# node_cal_list = 0:node_id 1:node_num 2:node_type 3:node_genP 4:node_cei 5:area_num                                                                                                          
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# transpose 转置，
E_N = np.dot(np.dot(np.linalg.inv(P_N - np.transpose(P_B)), np.transpose(P_G)), E_G)
epg = np.dot(np.linalg.inv(P_N - np.transpose(P_B)), np.transpose(P_G))
# print("np.dot(np.linalg.inv(P_N - np.transpose(P_B)), np.transpose(P_G))")
# print()

# print("E_G:", E_G)
# print("P_B:", P_B)
# print("P_N:", P_N)
# print("np.transpose(P_G)",np.transpose(P_G))
#print("E_G:",E_G)
print(len(E_N))
C_U = np.dot(np.transpose(E_N), P_L)

C_G = np.dot(np.transpose(E_G), P_G)  # 节点有功发电量*碳排因子
print("PG*EG")
print(np.dot(np.transpose(P_G),E_G))
"""
node_result_list = 0:node_id 1:node_num 2:node_cei 3:节点用功碳排放 4:节点发电碳排放 5:P_L节点用功消耗 6:area_num 7:节点有功发电 8:import_flag
# node_result_list = 0:node_id 1:node_num 2:CEI 3:CEU 4:CEG 5:P_L 6:area_num 
"""
node_result_list = []
number = 0
for index in node_cal_list:
    if (int(index[2]) == 3):  # 如果是变电站类型
        # 0:node_id,  1:node_num, E_N[number], C_U[0, number](猜测是节点用功碳排放), 0, P_L[number, number](节点用功量), area_num, 0, import_flag
        node_result_list.append(
            [index[0], index[1], E_N[number].tolist()[0], C_U[0, number].tolist(), 0, P_L[number, number], index[5], 0,
             index[7]])
    else:
        # 0:node_id,  1:node_num, node_cei, C_U[0, number](猜测是节点用功碳排放), C_G[0, number](节点发电碳排放), P_L[number, number](节点用功量), area_num, P_G[number, number](节点有功发电), import_flag
        node_result_list.append(
            [index[0], index[1], index[4], C_U[0, number].tolist(), C_G[0, number].tolist(), P_L[number, number],
             index[5], P_G[number, number], index[7]])
    number = number + 1

"""
line_result_list = 0:line_id 1:line_num 2:节点碳排因子node_cei(起始) 3:节点用功碳排/6 4:节点用功碳排
# line_result_list = 0:line_id 1:line_num 2:Line_CEI 3:Line_CEF 4:Line_CEFR
"""
line_result_list = []
number = 0
for line_data_item in line_data_list:
    for node_result_item in node_result_list:
        if line_data_item[2] == node_result_item[1]:  # 如果起始点
            temp1 = float(node_result_item[2]) * float(node_result_item[5]) / 6  # 节点碳排因子*P_L节点用功消耗/6 =节点用功碳排/6
            temp2 = float(node_result_item[2]) * float(node_result_item[5])  # 节点碳排因子*P_L节点用功消耗 = 节点用功碳排
            # 0:line_id 1:line_num 2:节点碳排因子node_cei(起始) 3:节点用功碳排/6 4:节点用功碳排
            line_result_list.append([line_data_item[0], line_data_item[1], node_result_item[2], temp1, temp2])
            number = number + 1

all_CEU = 0  # 节点用功碳排放
all_CEG = 0  # 节点发电碳排放
all_PL = 0  # P_L节点用功消耗
all_PG = 0  # 节点有功发电
all_CEI = 0  # 节点碳排因子node_cei(起始)
all_CFI = 0  # (猜测)节点发电碳排因子
for item in node_result_list:
    all_CEU = item[3] + all_CEU
    all_CEG = item[4] + all_CEG
    all_PL = item[5] + all_PL
    all_PG = item[7] + all_PG

if all_PL > 0:  # 如果P_L节点总用功消耗大于0
    all_CEI = all_CEU / all_PL  # 总碳排因子等于 总用功碳排/总用功消耗
else:
    print('error: 当前条件无法计算总碳排因子')

if all_PG > 0:  # 发电用功大于0
    all_CFI = all_CEG / all_PG  # 总发电碳排因子等于 总发电碳排/总有功发电
else:
    print('error: 当前条件无法计算总发电碳排因子')

print("all_CEU 节点用功碳排放:", all_CEU)
print("all_CEG 节点发电碳排放:", all_CEG)
print("all_PL P_L节点用功消耗:", all_PL)
print("all_PG 节点有功发电:", all_PG)
print("all_CEI 总碳排因子:", all_CEI)
print("all_CFI 总发电碳排因子:", all_CFI)
print("======================================")

import pandas as pd
from openpyxl import Workbook

# 假设你有如下变量
all_CEU = 23346.8736
all_CEG = 0.0
all_PL = 107.97101
all_PG = 48.16773
all_CEI = 216.23279804458622
all_CFI = 0.0

# 示例矩阵（确保它们是 numpy 数组或可以转换成列表）
# P_G, E_N, P_B, P_N 等变量应已定义

# 创建一个字典用于存储汇总信息
summary_data = {
    "指标": [
        "总节点用电碳排放 (all_CEU)",
        "总发电碳排放 (all_CEG)",
        "总节点用电消耗 (all_PL)",
        "总节点发电量 (all_PG)",
        "总碳排因子 (all_CEI)",
        "总发电碳排因子 (all_CFI)"
    ],
    "数值": [
        all_CEU,
        all_CEG,
        all_PL,
        all_PG,
        all_CEI,
        all_CFI
    ]
}

# 创建 DataFrame
df_summary = pd.DataFrame(summary_data)

# 将矩阵转为 DataFrame 列表
def matrix_to_df(matrix, name_prefix="Col"):
    """将二维矩阵转换为DataFrame"""
    return pd.DataFrame(matrix, columns=[f"{name_prefix}_{i}" for i in range(matrix.shape[1])]) if len(matrix.shape) > 1 else pd.DataFrame(matrix.reshape(-1, 1), columns=[name_prefix])

df_E_N = matrix_to_df(E_N, "E_N")
df_P_B = matrix_to_df(P_B, "P_B")
df_P_N = matrix_to_df(P_N, "P_N")
df_P_G_T = matrix_to_df(np.transpose(P_G), "P_G")  # 转置后按列展开


# 写入 Excel 文件
output_file = '碳因子计算结果.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_summary.to_excel(writer, sheet_name='汇总', index=False)
    df_E_N.to_excel(writer, sheet_name='E_N', index=False)
    df_P_B.to_excel(writer, sheet_name='P_B', index=False)
    df_P_N.to_excel(writer, sheet_name='P_N', index=False)
    df_P_G_T.to_excel(writer, sheet_name='P_G', index=False)

print(f"结果已保存至 {output_file}")