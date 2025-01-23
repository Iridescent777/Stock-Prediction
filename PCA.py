import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np

from sklearn.decomposition import PCA

# 读取数据
df = pd.read_csv(r"E:\\云盘重要文件\\西安交通大学\\辅修\\Python\\forecast_original_sample.csv", header = None)

df = pd.DataFrame(df)

X = df.iloc[:,:-1]

x_Mean = X.mean()

x_Std = X.std()

Z = (X - x_Mean) / x_Std

c = Z.cov()


eigenvalues , eigenvectors = np.linalg.eig(c)

print('eigenvalues: \n', eigenvalues)

print('eigenvalues shape: \n', eigenvalues.shape)

print('eigenvectors Shape: \n', eigenvectors.shape)

idx = eigenvalues.argsort()[::-1]

eigenvalues = eigenvalues[idx]

eigenvectors = eigenvectors[:,idx]

explained_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

print('explained_variance: \n', explained_variance)

n_components = np.argmax(explained_variance >= 0.95) + 1

print('n_components: \n', n_components)

n_components += 2

u = eigenvectors[:,:n_components]
pca_component = pd.DataFrame(u,
                             columns = ['PC1','PC2', 'PC3', 'PC4', 'PC5', 'PC6','PCA7','PCA8','PCA9','PCA10', 'PCA11', 'PCA12']
                            )


pca = PCA(n_components=n_components)

pca.fit(Z)

explained_variance_ratio = pca.explained_variance_ratio_

print('explained_variance_ratio: \n', explained_variance_ratio)

pca_components = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5','PC6', 'PC7', 'PC8', 'PC9', 'PC10','PCA11','PCA12'],index = X.columns + 1)

cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

for i, ratio in enumerate(explained_variance_ratio, 1):

    print(f'PC{i} explained variance ratio: {ratio:.2%}')

print('Cumulative explained variance ratio:')

for i, ratio in enumerate(cumulative_variance_ratio, 1):

    print(f'PC{i} cumulative explained variance ratio: {ratio:.2%}')

# 绘制累积解释方差比例图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Primary Components')
plt.ylabel('Cmulative Explained Variance Ratio')
plt.title('PCA Cumulative Explained Variance Ratio')
plt.grid(True)

# 在图中添加数值标签
for i, ratio in enumerate(cumulative_variance_ratio):
    plt.annotate(f'{ratio:.2%}', (i + 1, ratio), textcoords="offset points", xytext=(0,10), ha='center')

# 添加95%的参考线
plt.axhline(y=0.95, color='r', linestyle='--', label='95% \\Explained Variance')
plt.legend()

# 保存图片
plt.savefig('E:\\云盘重要文件\\西安交通大学\\辅修\\Python\\PCA_cumulative_variance.png')
plt.show()

# print('pca.components_: \n', pca_components)

# pca_components.to_csv('E:\\云盘重要文件\\西安交通大学\\辅修\\Python\\PCA_Weight.csv',index = True, header = False)
# # Matrix multiplication or dot Product
Z_pca = Z @ pca_component
# # Rename the columns name
Z_pca.rename({'PC1': 'PCA1', 'PC2': 'PCA2', 'PC3': 'PCA3', 'PC4': 'PCA4', 'PC5': 'PCA5', 'PC6': 'PCA6', 'PC7': 'PCA7',
'PC8': 'PCA8', 'PC9':'PCA9', 'PC10': 'PCA10','PC11': 'PCA11', 'PC12':'PCA12'}, axis=1, inplace=True)


Z_pca['Label'] = df.iloc[:,-1].reset_index(drop = True)

print(Z_pca)

# Print the  Pricipal Component values
Z_pca.to_csv('E:\\云盘重要文件\\西安交通大学\\辅修\\Python\\Test_PCA.csv',index = True, header = False)
