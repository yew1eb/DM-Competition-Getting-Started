from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition
import numpy as np

PCA_COMPONENTS = 100


def doWork(train, labels, test):
    print("Converting training set to matrix")
    X_train = np.mat(train)

    print("Fitting PCA. Components: %d" % PCA_COMPONENTS)
    pca = decomposition.PCA(n_components=PCA_COMPONENTS).fit(X_train)

    print("Reducing training to %d components" % PCA_COMPONENTS)
    X_train_reduced = pca.transform(X_train)

    print("Fitting kNN with k=10, kd_tree")
    knn = KNeighborsClassifier(n_neighbors=10, algorithm="kd_tree")
    print(knn.fit(X_train_reduced, labels))

    print("Reducing test to %d components" % PCA_COMPONENTS)
    X_test_reduced = pca.transform(test)

    print("Preddicting numbers")
    predictions = knn.predict(X_test_reduced)

    print("Writing to file")
    write_to_file(predictions)

    return predictions


def write_to_file(predictions):
    f = open("output-pca-knn-skilearn-v3.csv", "w")
    for p in predictions:
        f.write(str(p))
        f.write("\n")
    f.close()


if __name__ == '__main__':
    from load_data import read_data
    train, labels = read_data("../data/train.csv")
    test, tmpl = read_data("../data/test.csv", test=True)
    print(doWork(train, labels, test))
