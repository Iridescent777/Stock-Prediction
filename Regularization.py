import pandas as pd
import numpy as np

# 读取数据
PTSX0 = pd.read_csv('train_original_sample.csv', header=None)
forecast_sample = pd.read_csv('forecast_original_sample.csv', header=None)

# 训练样本归一化
sxn1, sxm1 = PTSX0.shape
S_X_T = pd.DataFrame(index=PTSX0.index, columns=PTSX0.columns)
S_X_T.iloc[:, 0] = PTSX0.iloc[:, 0]
S_X_T.iloc[:, 20] = PTSX0.iloc[:, 20]  # Python索引从0开始，所以减1

for k in range(0, sxm1 - 1):
    xm2 = PTSX0.iloc[:, k].mean()
    std2 = PTSX0.iloc[:, k].std()
    for j in range(len(PTSX0)):
        if PTSX0.iloc[j, k] > xm2 + 2 * std2:
            S_X_T.iloc[j, k] = 1
        elif PTSX0.iloc[j, k] < xm2 - 2 * std2:
            S_X_T.iloc[j, k] = 0
        else:
            S_X_T.iloc[j, k] = (PTSX0.iloc[j, k] - (xm2 - 2 * std2)) / (4 * std2)

# 保存训练样本文件
S_X_T.to_csv('train_sample.csv', index=False, header=False)

# 预测样本归一化
sxn2, sxm2 = forecast_sample.shape
S_X_F = pd.DataFrame(index=forecast_sample.index, columns=forecast_sample.columns)
S_X_F.iloc[:, 0] = forecast_sample.iloc[:, 0]

for k in range(0, sxm2):
    xm2 = forecast_sample.iloc[:, k].mean()
    std2 = forecast_sample.iloc[:, k].std()
    for j in range(len(forecast_sample)):
        if forecast_sample.iloc[j, k] > xm2 + 2 * std2:
            S_X_F.iloc[j, k] = 1
        elif forecast_sample.iloc[j, k] < xm2 - 2 * std2:
            S_X_F.iloc[j, k] = 0
        else:
            S_X_F.iloc[j, k] = (forecast_sample.iloc[j, k] - (xm2 - 2 * std2)) / (4 * std2)

# 保存归一化之后的预测样本文件
S_X_F.to_csv('forecast_sample.csv', index=False, header=False)