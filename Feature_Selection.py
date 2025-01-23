import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取变量信息
tdata = pd.read_csv('train_sample.csv', header=None)
fdata = pd.read_csv('forecast_sample.csv', header=None)
rn=21
cn=21

# 计算并显示相关系数矩阵
A = tdata.iloc[:, 1:]
covmat = np.corrcoef(A.T)  # 转置A以计算列之间的相关性

# 设置变量名称
varargin = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
           'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'y']

# 绘制相关系数矩阵热图
plt.figure()
plt.imshow(covmat, cmap='coolwarm')
plt.xticks(np.arange(len(varargin)), varargin)
plt.yticks(np.arange(len(varargin)), varargin)
plt.colorbar()
plt.show()

# 选择相关性较强的变量
covth = 0.2
c1 = covmat[cn-2,0:cn-3]  # 获取最后一列的相关系数，排除了因变量y
print(c1)
vid = np.abs(c1) > covth
print(vid)
A1=A.iloc[:, 0:(cn-3)];
A2=A1.iloc[:, vid]  # 选择相关性较强的变量
stdata = pd.concat([tdata.iloc[:, 0], A2, tdata.iloc[:, -1]], axis=1)

B = fdata.iloc[:, 1:-1]  # 排除第一列和最后一列
B1 = B.iloc[:, vid]  # 选择相关性较强的变量
sfdata = pd.concat([fdata.iloc[:, 0], B1], axis=1)

# 保存选中的数据为Excel文件
stdata.to_csv('selected_tdata.csv', index=False, header=False)
sfdata.to_csv('selected_fdata.csv', index=False, header=False)