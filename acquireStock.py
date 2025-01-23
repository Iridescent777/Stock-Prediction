import pandas as pd
import time
from tqdm import tqdm

# 读取 CSV 文件
df1 = pd.read_csv(r"E:\\云盘重要文件\\西安交通大学\\辅修\\Python\\TRD_Dalyr.csv")
df2 = pd.read_csv(r"E:\\云盘重要文件\\西安交通大学\\辅修\\Python\\TRD_Dalyr1.csv")

# 打印 'Stkcd' 列的数据类型
if 'Stkcd' in df1.columns:
    print(df1['Stkcd'].dtypes)
else:
    print("Column 'Stkcd' not found in df1")


# 初始化计数器和当前值
count = 0
curr = None
selected_rows = []

# 遍历 DataFrame 的每一行
for index, row in df1.iterrows():

    if row['Stkcd'] != curr and curr is not None:
        # 当 Stkcd 变化时，保存之前的数据到 Excel 文件
        # 当前excel文件计数+1
        count += 1
        selected_df1 = pd.DataFrame(selected_rows)
        stkcd_str = f"{int(count):06d}"  # 高位补零，长度为6
        file_path = rf"E:\\云盘重要文件\\西安交通大学\\辅修\\Python\\TRD_Dalyr\\{stkcd_str}.xlsx"
            
        with tqdm(total=len(selected_df1)) as pbar:
            for i, r in selected_df1.iterrows():
                pbar.update(1)
            
        selected_df1.to_excel(file_path, index=False)
        selected_rows.clear()

    # 更新行的 'Stkcd' 值
    curr = row['Stkcd']
    # 将日期转换为时间戳
    row['Trddt'] = time.mktime(time.strptime(str(row['Trddt']), '%Y-%m-%d'))
    selected_rows.append(row)

    if count == 2000:
        break
