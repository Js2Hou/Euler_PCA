import cmath
from functools import reduce

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ComplexPCA(object):
    """Complex principle component analysis

    Input
    --------
    X : array-like of shape (n_samples, n_features)
        X is a complex-valued matrix

    Parameter
    --------
    n_components : int or None
        Numbers of components to keep.
        if n_components is not set, then all components are kept:
            n_components == min(n_samples, n_features)

    copy : bool, default=True
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    Attributes
    --------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        ``explained_variance_``.

    explained_variance_ : array, shape (n_components,)

    explained_variance_ratio_ : array, shape (n_components,)
    """

    def __init__(self, n_components=None, *, copy=True):
        self.explained_variance_ratio_ = []
        self.n_components = n_components
        self.copy = copy
        self.mmscale = MinMaxScaler()

    def data_process(self, X, y=None):
        return self.mmscale.transform(X)

    def inverse_data_process(self, X, y=None):
        return self.mmscale.inverse_transform(X)

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        --------
        X : array_like, shape (n_samples, n_features)
            Training data, where n_samples if the number of samples
            and n_features if the number of features

        y : None
            Ignored variable

        Returns
        --------
        self : object
            Returns the instance itself.
        """
        if self.copy is True:
            X = X.copy()
        self.mmscale.fit(X)
        X = self.data_process(X)
        self._fit(X)
        return self

    def _fit(self, Z):
        """Fit the model by computing eigenvalue decomposition on X * X.H"""

        # Handle n_components==None
        if self.n_components is None:
            n_components = min(Z.shape)
        else:
            n_components = self.n_components

        n_samples, n_features = Z.shape

        # if self.copy is True:
        #     Z = Z.copy()

        # eigenvalue decomposition method
        K = np.dot(Z, np.conj(Z).T)
        w, v = np.linalg.eigh(K)
        w1 = np.flip(w)  # one-dimension
        v1 = np.fliplr(v)
        B = reduce(np.dot, [np.conj(Z).T, v1[:, :n_components], np.diag(np.float_power(w1[:n_components], -0.5))])
        B_H = np.conj(B).T

        components_ = B_H

        # Get variance explained by singular values
        try:
            # bias 1 : void numerical overflow; ensure function return a non-negative value
            explained_variance_ = np.log(w1 + 1)
            total_var = explained_variance_.sum()
            explained_variance_ratio_ = explained_variance_ / total_var
        except ZeroDivisionError:
            return ZeroDivisionError

        threshold = 0.8
        explained_variance_ratio_sum_ = 0
        for i, e in enumerate(explained_variance_ratio_):
            explained_variance_ratio_sum_ += e
            if explained_variance_ratio_sum_ >= threshold:
                n_components = i + 1
                break

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.n_components_ = components_[:n_components]
        self.n_components = n_components
        self.explained_variance_ratio = explained_variance_ratio_[:n_components]
        self.explained_variance_ratio_sum = explained_variance_ratio_sum_
        self.B_H, self.B = B_H, B

        return B_H, B

    def transform(self, Z):
        Z = Z.copy()
        Z = self.data_process(Z)
        # print(f'B_H : {self.B_H.shape}, Z : {Z.shape}')
        return (self.B_H @ Z.T).T

    def inverse_transform(self, Y):
        Y = Y.copy()
        # return np.dot(self.B, X)
        X = (self.B @ Y.T).T
        return self.inverse_data_process(X)


class EPCA(ComplexPCA):
    def __init__(self, alpha=1.9):
        super().__init__()
        self.alpha = alpha
        self.mmscale = MinMaxScaler()

    def data_process(self, X, y=None):
        X = self.mmscale.transform(X)
        return self.euler(X)

    def inverse_data_process(self, X, y=None):
        X = self.inverse_euler(X)
        return self.mmscale.inverse_transform(X)

    def euler(self, x):
        alpha = self.alpha
        z = np.eye(*x.shape, dtype=complex)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i, j] = (1 / cmath.sqrt(2)) * (cmath.exp(1j * alpha * cmath.pi * x[i, j]))
        return z

    def inverse_euler(self, z):
        alpha = self.alpha
        x = np.eye(*z.shape)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                x[i, j] = (cmath.atan(z[i, j].imag / z[i, j].real) / (alpha * cmath.pi)).real
        return x


class KNN(object):
    """k-nearest neighbor

    """

    def __init__(self, train_x, test_x, train_labels, test_labels, k):
        self.train_x = train_x
        self.test_x = test_x
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.k = k

    def fit(self):
        train_x = self.train_x.copy()
        test_x = self.test_x.copy()
        train_labels = self.train_labels
        test_labels = self.test_labels
        n = test_x.shape[0]
        right_num = 0
        pred_labels = []

        for id, y in enumerate(test_x):
            distance = np.array([np.linalg.norm(i - y) for i in train_x])
            distance_map_label = np.vstack((distance, train_labels)).T
            distance_map_label = distance_map_label[np.argsort(distance_map_label[:, 0])]
            distance_map_label = distance_map_label.astype('int64')
            label_pred = np.argmax(np.bincount(distance_map_label[:self.k, 1]))
            pred_labels.append(label_pred)
            if label_pred == test_labels[id]:
                right_num += 1
        accuracy = right_num / n
        pred_labels = np.array(pred_labels)
        return accuracy, pred_labels, test_labels
