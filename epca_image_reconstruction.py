from operator import itemgetter

import numpy as np
from scipy.io import loadmat

from models import ComplexPCA, EPCA
from utils import plot


def load_data(path=r'./data/data.mat'):
    dataset = loadmat(path)
    b = itemgetter('train_X', 'train_Y', 'test_X', 'test_Y')
    X = b(dataset)[0]
    return X[:50, :]


train_X = load_data()

# 训练模型
n_components = 10
pca = ComplexPCA(n_components)
pca.fit(train_X)
print(f'PCA累积方差对数贡献率为：{pca.explained_variance_ratio_sum}')

epca = EPCA(n_components)
epca.fit(train_X)
print(f'EPCA累积方差对数贡献率为：{epca.explained_variance_ratio_sum}')

# ----------
# 图像重建
x_true = train_X[0, :]

y_pca = pca.transform(x_true[np.newaxis, :])
x_pred_pca = pca.inverse_transform(y_pca)

y_epca = epca.transform(x_true[np.newaxis, :])
x_pred_epca = epca.inverse_transform(y_epca)

# 显示结果
print(f'pca真实与预测图像的像素差范数：{np.linalg.norm(x_true - x_pred_pca)}')
print(f'epca真实与预测图像的像素差范数：{np.linalg.norm(x_true - x_pred_epca)}')

# 绘图
plot([x_true.reshape(84, 96).T, x_pred_pca.reshape(84, 96).T, x_pred_epca.reshape(84, 96).T], 3,
     ['true', 'pca', 'epca'])
