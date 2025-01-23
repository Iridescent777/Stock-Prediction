import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置字体为SimHei（或其他已安装的中文字体）
mpl.rcParams['font.family'] = 'SimHei'  # 替换为你系统中可用的中文字体
mpl.rcParams['axes.unicode_minus'] = False  # 处理负号显示问题
# 读取数据，确保使用正确的函数
data = pd.read_excel('Train_PCA.xlsx')  # 使用 read_excel 读取 Excel 文件
# 分离特征和标签
X = data.iloc[:, 1:12]  # 特征
y = data.iloc[:, 13]     # 标签

# 划分数据集，80%用于训练，20%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=10)

# 创建支持向量机分类器实例并训练模型
clf = SVC(probability=True, random_state=42)  # probability=True 允许我们获取预测概率
clf.fit(X_train, y_train)

# 对测试样本进行预测
y_test_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_test_pred)

# 计算ROC曲线和AUC值
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

# 输出结果
print(f'模型准确率: {accuracy:.2f}')
print("训练样本的原始标签向量:")
print(y_train.values)
print("模型对测试样本的标签计算结果:")
print(y_test_pred)
print(f'AUC值: {roc_auc:.2f}')

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC曲线 (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC曲线')
plt.legend(loc='lower right')
plt.show()

# 保存预测结果到Excel文件
results_df = pd.DataFrame({
    '原始标签': y_test.values,
    '预测标签': y_test_pred
})

# 确保已经安装了openpyxl库
results_df.to_excel('train_result_SVM.xlsx', index=False, engine='openpyxl')


# 读取数据，确保使用正确的函数
data0 = pd.read_excel('Test_PCA.xlsx')  # 使用 read_excel 读取 Excel 文件

# 分离特征和标签
X0 = data0.iloc[:, 1:12]  # 特征
x0_test = data0.iloc[:, 0]

y0_test_pred = clf.predict(X0)
# 保存预测结果到Excel文件

results_df = pd.DataFrame({
    '原始标签': x0_test,
    '预测标签': y0_test_pred
})
results_df.to_excel('forecast_result_SVM.xlsx', index=False, engine='openpyxl')