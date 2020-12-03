from load_data import load_data
from models import EPCA, KNN

if __name__ == '__main__':
    # load data
    train_x, train_y, test_x, test_y = load_data(ratio=0.9, shuffle_=True)

    # euler-pca fit
    epca = EPCA(3)
    epca.fit(train_x)
    print(f'epca explained variance ratio : {epca.explained_variance_ratio_sum}')

    # euler-pca transform
    train_x = epca.transform(train_x)
    test_x = epca.transform(test_x)

    # image classification by knn
    k = 5
    knn = KNN(train_x, test_x, train_y, test_y, k)
    acc, pred_labels, test_labels = knn.fit()
    for i, j in zip(pred_labels, test_labels):
        print(f'predicted value : {i} true value : {j}')

    print(f'acc : {acc}')
