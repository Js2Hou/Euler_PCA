import os

import numpy as np
from scipy.io import loadmat, savemat
from sklearn.utils import shuffle


def load_data(datasets_path=r'./data/', shuffle_=True, ratio=0.8):
    """
    Returns
    --------
    train_X : array-like of shape (n_samples, n_features)
    train_Y : array-like of shape (n_shamples, )
    test_X :
    test_Y :
    """
    list_mat_path = [i for i in os.listdir(datasets_path) if i.startswith('ExtYaleB')]  # 包含5个.mat文件
    data_list = []
    for i in list_mat_path:
        mat = loadmat(os.path.join(datasets_path, i))
        data = mat['DAT']
        data_list.append(data)

    data1 = np.concatenate(data_list, axis=1)  # (8064, 64, 38)
    data_list1 = []
    _, a, b = data1.shape
    for i in range(a):
        for j in range(b):
            data_list1.append(data1[:, i, j])

    X = np.array(data_list1)
    Y = np.hstack([[i for j in range(64)] for i in range(38)])

    if shuffle_ is True:
        X, Y = shuffle(X, Y)

    # X = X[:500, :]
    # Y = Y[:500]

    n = X.shape[0]
    split_loc = int(n * ratio)
    train_x, train_y, test_x, test_y = X[:split_loc, :], Y[:split_loc], X[split_loc + 1:, :], Y[split_loc + 1:]

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_data()
    mdict = {'train_X': train_X, 'train_Y': train_Y, 'test_X': test_X, 'test_Y': test_Y}
    savemat(r'./data/data.mat', mdict=mdict)
