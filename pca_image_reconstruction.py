import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA

# ----------
# 数据处理
# ----------
path = r'data/masked_faces.mat'
# shape = 32, 32, 100
data = loadmat(path)['Test_facemask'][:, :, 0, :]
shape = data.shape
# shape = 32 * 32, 100
data_flat = data.reshape((shape[0] * shape[1], shape[2]))
training_data = data_flat.transpose((1, 0))
# ----------
# 变换
pca = PCA(1)
pca.fit(training_data)
print(pca.explained_variance_ratio_)
print(pca.get_params())

x = training_data[0, :][np.newaxis, :]  # (1, 1024)
print(x.shape)

y = pca.transform(x)  # (1, 1)

image_ori = x.reshape(shape[:-1])
x_from_y = pca.inverse_transform(y)
image_transformed = x_from_y.reshape(shape[:-1])

# ----------
# 绘图
plt.figure()
plt.subplot(121)
plt.imshow(image_ori)
plt.title('original')

plt.subplot(122)
plt.imshow(image_transformed)
plt.title('reformulated')

plt.show()
