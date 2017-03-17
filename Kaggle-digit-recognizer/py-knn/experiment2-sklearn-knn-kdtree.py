import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def doWork(train, test, labels):
    print("Converting training to matrix")
    train_mat = np.mat(train)
    print("Fitting knn")
    knn = KNeighborsClassifier(n_neighbors=10, algorithm="kd_tree")
    print(knn.fit(train_mat, labels))
    print("Preddicting")
    predictions = knn.predict(test)
    print("Writing to file")
    write_to_file(predictions)
    return predictions


def write_to_file(predictions):
    f = open("output-knn-skilearn.csv", "w")
    for p in predictions:
        f.write(str(p))
        f.write("\n")
    f.close()


if __name__ == '__main__':
    from load_data import read_data
    train, labels = read_data("../data/train.csv")
    test, tmpl = read_data("../data/test.csv", test=True)
    predictions = doWork(train, test, labels)
    print(predictions)
