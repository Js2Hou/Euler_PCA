import cmath

import matplotlib.pyplot as plt
import numpy as np


def plot(imgs, col, titles=None):
    if titles is None:
        titles = []
    img_num = len(imgs)
    row = (img_num + col - 1) // col

    if len(titles) == 0:
        titles = [f'img {i}' for i in range(img_num)]

    plt.figure()
    for i in range(img_num):
        plt.subplot(row, col, i + 1)
        plt.imshow(imgs[i])
        plt.title(titles[i])
    plt.show()


def euler(x, alpha=1.9):
    z = np.eye(*x.shape, dtype=complex)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i, j] = (1 / cmath.sqrt(2)) * (cmath.exp(1j * alpha * cmath.pi * x[i, j]))
    return z


def euler_inv(z, alpha=1.9):
    x = np.eye(*z.shape)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            x[i, j] = (cmath.atan(z[i, j].imag / z[i, j].real) / (alpha * cmath.pi)).real
    return x


if __name__ == '__main__':
    a = np.arange(12).reshape(3, 4)
    b = euler(a)
