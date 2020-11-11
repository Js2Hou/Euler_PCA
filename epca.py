from scipy.io import loadmat, savemat
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import math

# ----------
# 变量设置
alpha = 1.9

e = math.e
pi = math.pi

# ----------
# 数据处理
# ----------
path = r'data/data.mat'
# shape = 32, 32, 100
data = loadmat(path)['Test_facemask'][:, :, 0, :]
shape = data.shape
# shape = 32 * 32, 100
data_flat = data.reshape((shape[0] * shape[1], shape[2]))
training_data = data_flat.transpose((1, 0))  # (100, 1024)
z = np.eye(*training_data.shape)
for i in range(training_data.shape[0]):
    z[i] = 1 / math.sqrt(2) * e ** (1j * alpha * pi * training_data[i])
print(z.shape)
# ----------
# 变换
pca = PCA(100)
pca.fit(z)
print(pca.explained_variance_ratio_)
print(pca.get_params())

x = z[0, :][np.newaxis, :]  # (1, 1024)
print(x.shape)

y = pca.transform(x)  # (1, 1)

image_ori = x.reshape(shape[:-1])
z_from_y = pca.inverse_transform(y)
x_from_z = z_from_y / (alpha * pi)
image_transformed = x_from_z.reshape(shape[:-1])

# ----------
# 绘图
plt.figure()
plt.subplot(121)
plt.imshow(image_ori)
plt.title('original')

plt.subplot(122)
plt.imshow(image_transformed)
plt.title('transformed')

plt.show()
