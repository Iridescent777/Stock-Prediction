import os
import pandas as pd
import numpy as np
def mysum(P, h, k, t):
    sum = 0
    for i in range(h,k):

        sum += P[i][t]

    return sum

def myMean(P, h, k, t):

    sum = 0

    for i in range(h,k):

        sum += P[i][t]

    return sum / (k-h)

# 参数定义
train_s1 = []
stn = 0  # 股票总个数
train_num = 0  # 训练样本记录条数
forecast_num = 0  # 预测样本记录条数
good_s_n = 0  # 好股票记录个数
bad_s_n = 0  # 坏股票记录个数
common_s_n = 0  # 一般股票记录个数

# 数据文件夹路径
dirname = 'data'
files = [f for f in os.listdir(dirname) if f.endswith('.xlsx')]
SN = len(files)
forecast_sample = []

for file in files:
    # 读取 Excel 文件
    filepath = os.path.join(dirname, file)
    P = pd.read_excel(filepath, header=None).values

    # 删除成交量为 0 的行
    m=len(P)
    n=len(P[0])

    ii=0
    for iii in range(m):
        if P[ii][5]==0:
            P[ii][:]=[]
        else:
            ii+=1


    # 删除开盘有效天数少于 120 的股票
    if len(P) < 120:
        continue

    # 记录有效股票数量
    stn += 1

    for h in range(1,21):
        # 跳过第 1 和第 2 次迭代
        if h in [2, 3]:
            continue

        # 计算各指标
        if h + 30 >= len(P):
            break  # 防止索引越界

        s_x1 = 100 * (P[h][4] - P[h + 1][4]) / P[h + 1][4]
        s_x2 = 100 * (P[h][4] - P[h + 2][4]) / P[h + 2][4]
        s_x3 = 100 * (P[h][4] - P[h + 5][4]) / P[h + 5][4]
        s_x4 = 100 * (P[h][4] - P[h + 10][4]) / P[h + 10][4]
        s_x5 = 100 * (P[1][4] - P[h + 30][4]) / P[h + 30][4]

        # 10 日涨跌比率 ADR 和 RSI
        rise_num=0
        dec_num=0
        for j in range(1,11):
            rise_rate=100*(P[h+j-1][4]-P[h+j][4])/P[j+h][4];
            if rise_rate >= 0:
                rise_num = rise_num + 1;
            else:
                dec_num = dec_num + 1;
        s_x6 = rise_num / (dec_num + 0.01)
        s_x7 = rise_num / 10

        # K 线值和均值
        s_kvalue = [(P[h + j - 1][4] - P[h + j - 1][1]) / (P[h + j - 1][2] - P[h + j - 1][3] + 0.01) for j in range(1,7)]
        s_x8 = s_kvalue[0]
        s_x9 = np.mean(s_kvalue[:3])
        s_x10 = np.mean(s_kvalue[:6])

        # BIAS 指标
        s_x11 = (P[h][4] - mysum(P,h,h+6,4)/6) / (mysum(P, 1, h+6, 4)/6)
        s_x12 = (P[h][4] - mysum(P,h,h+10,4)/10) / (mysum(P, 1, h+10, 4)/10)

        # RSV 指标
        minp=99999999999
        maxp=-99999
        for j in range(1,h+9):
            if(P[j][4]>maxp):
                maxp=P[j][4]
            if (P[j][4] < minp):
                minp = P[j][4]
        s_x13 = (P[h][4] - minp) / (maxp - minp)
        minp = 99999999999
        maxp = -99999
        for j in range(1, h + 30):
            if (P[j][4] > maxp):
                maxp = P[j][4]
            if (P[j][4] < minp):
                minp = P[j][4]
        s_x14 = (P[h][4] - minp) / (maxp - minp)
        minp = 99999999999
        maxp = -99999
        for j in range(1, h + 90):
            if (P[j][4] > maxp):
                maxp = P[j][4]
            if (P[j][4] < minp):
                minp = P[j][4]
        s_x15 = (P[h][4] - minp) / (maxp - minp)

        # OBV 指标

        s_x16 = np.sign(P[h][4] - P[h + 1][4]) * P[h][5] / (mysum(P, h, h+5, 5) / 5)

        OBV_5 = np.sum([np.sign(P[h + j - 1][4] - P[h + j][4]) * P[h + j - 1][5] for j in range(1, 6)])
        OBV_10 = np.sum([np.sign(P[h + j - 1][4] - P[h + j][4]) * P[h + j - 1][5] for j in range(1, 11)])
        OBV_30 = np.sum([np.sign(P[h + j - 1][4] - P[h + j][4]) * P[h + j - 1][5] for j in range(1, 31)])
        OBV_60 = np.sum([np.sign(P[h + j - 1][4] - P[h + j][4]) * P[h + j - 1][5] for j in range(1, 61)])

        s_x17 = OBV_5 / (mysum(P, h,h+5, 5) / 5)
        s_x18 = OBV_10 / (mysum(P, h,h+5, 5) / 5)
        s_x19 = OBV_30 / (mysum(P, h,h+5, 5) / 5)
        s_x20 = OBV_60 / (mysum(P, h,h+5, 5) / 5)

        if h == 1:
            forecast_num += 1
            # forecast_sample.append([int(file[2:8]), s_x1, s_x2, s_x3, s_x4, s_x5, s_x6, s_x7, s_x8, s_x9, s_x10,
            #                          s_x11, s_x12, s_x13, s_x14, s_x15, s_x16, s_x17, s_x18, s_x19, s_x20])
            forecast_sample.append([s_x1, s_x2, s_x3, s_x4, s_x5, s_x6, s_x7, s_x8, s_x9, s_x10,
                                     s_x11, s_x12, s_x13, s_x14, s_x15, s_x16, s_x17, s_x18, s_x19, s_x20])
            continue

        # 判断好坏股票
        s_y = 0
        rise_1 = 100 * (P[h - 1][4] - P[h][4]) / P[h][4]
        rise_2 = 100 * (P[h - 3][4] - P[h][4]) / P[h][4]

        if rise_1 >= 4 and rise_2 >= 6:
            s_y = 1
            good_s_n += 1
        elif rise_1 < 0 and rise_2 < 0:
            s_y = -1
            bad_s_n += 1
        else:
            common_s_n += 1

        train_num += 1
        # train_s1.append([int(files[2:8]), s_x1, s_x2, s_x3, s_x4, s_x5, s_x6, s_x7, s_x8, s_x9, s_x10, s_x11,
        #                  s_x12, s_x13, s_x14, s_x15, s_x16, s_x17, s_x18, s_x19, s_x20, s_y])
        train_s1.append([s_x1, s_x2, s_x3, s_x4, s_x5, s_x6, s_x7, s_x8, s_x9, s_x10, s_x11,
                         s_x12, s_x13, s_x14, s_x15, s_x16, s_x17, s_x18, s_x19, s_x20, s_y])

# 挑选样本
part_num = min([good_s_n, bad_s_n, common_s_n])
g_sample, c_sample, b_sample = [], [], []

for record in train_s1:
    if record[20] == 1 and len(g_sample) < part_num:
        g_sample.append(record)
    elif record[20] == 0 and len(c_sample) < part_num:
        c_sample.append(record)
    elif record[20] == -1 and len(b_sample) < part_num:
        b_sample.append(record)

PTSX0 = g_sample + b_sample
if not PTSX0:
    print("没有符合条件的数据样本")
else:
    # 保存训练样本和预测样本
    print(pd.DataFrame(forecast_sample))
    pd.DataFrame(PTSX0).to_csv('./train_original_sample.csv', index=False, header=False)
    pd.DataFrame(forecast_sample).to_csv('./forecast_original_sample.csv', index=False, header=False)
