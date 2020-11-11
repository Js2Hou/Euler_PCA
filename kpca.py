from scipy.io import loadmat, savemat
from sklearn.decomposition import KernelPCA
import numpy as np
import matplotlib.pyplot as plt

# ----------
# 数据处理
# ----------
path = r'data/data.mat'
# shape = 32, 32, 100
data = loadmat(path)['Test_facemask'][:, :, 0, :]
shape = data.shape
# shape = 32 * 32, 100
data_flat = data.reshape((shape[0] * shape[1], shape[2]))
training_data = data_flat.transpose((1, 0))
# ----------
# 变换
kpca = KernelPCA(n_components=1, kernel='cosine', fit_inverse_transform=True)
kpca.fit(training_data)
print(kpca.get_params())
# print(kpca.explained_variance_ratio_)
x = training_data[0, :][np.newaxis, :]  # (1, 1024)
print(x.shape)

y = kpca.transform(x)  # (1, 100)

x = x[:, np.newaxis]
image_ori = x.reshape(shape[:-1])
x_from_y = kpca.inverse_transform(y)
image_transformed = x_from_y.reshape(shape[:-1])

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
