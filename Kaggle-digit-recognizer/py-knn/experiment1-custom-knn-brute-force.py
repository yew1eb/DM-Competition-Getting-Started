import numpy as np
import operator
import time


# euclidean distance without square root to save
# some computational time
def euclid(x1, x2):
    return np.sum(np.power(np.subtract(x1, x2), 2))


# calc kNN from test_row vs training set
# default k = 5
# brute force! :(
def knn(test_row, train, k=5):
    diffs = {}
    idx = 0
    start = time.time()
    for t in train:
        diffs[idx] = euclid(test_row, t)
        idx = idx + 1
    print("for loop: %f idx(%d)" % (time.time() - start, idx))
    return sorted(iter(diffs.items()), key=operator.itemgetter(1))[:k]


# majority vote
def majority(knn, labels):
    a = {}
    for idx, distance in knn:
        if labels[idx] in list(a.keys()):
            a[labels[idx]] = a[labels[idx]] + 1
        else:
            a[labels[idx]] = 1
    return sorted(iter(a.items()), key=operator.itemgetter(1), reverse=True)[0][0]


# worker. crawl through test set and predicts number
def doWork(train, test, labels):
    output_file = open("output.csv", "w", 0)
    idx = 0
    size = len(test)
    for test_sample in test:
        idx += 1
        start = time.time()
        prediction = majority(knn(test_sample, train, k=100), labels)
        print("Knn: %f" % (time.time() - start))
        output_file.write(prediction)
        output_file.write("\n")
        print((float(idx) / size) * 100)
    output_file.close()


# majority vote for a little bit optimized worker
def majority_vote(knn, labels):
    knn = [k[0, 0] for k in knn]
    a = {}
    for idx in knn:
        if labels[idx] in list(a.keys()):
            a[labels[idx]] = a[labels[idx]] + 1
        else:
            a[labels[idx]] = 1
    return sorted(iter(a.items()), key=operator.itemgetter(1), reverse=True)[0][0]


def doWorkNumpy(train, test, labels):
    k = 20
    train_mat = np.mat(train)
    output_file = open("output-numpy2.csv", "w", 0)
    idx = 0
    size = len(test)
    for test_sample in test:
        idx += 1
        start = time.time()
        knn = np.argsort(np.sum(np.power(np.subtract(train_mat, test_sample), 2), axis=1), axis=0)[:k]
        s = time.time()
        prediction = majority_vote(knn, labels)
        output_file.write(prediction)
        output_file.write("\n")
        print("Knn: %f, majority %f" % (time.time() - start, time.time() - s))
        print("Done: %f" % (float(idx) / size))
    output_file.close()
    output_file = open("done.txt", "w")
    output_file.write("DONE")
    output_file.close()


if __name__ == '__main__':
    from load_data import read_data
    train, labels = read_data("../data/train.csv")
    test, tmpl = read_data("../data/test.csv", test=True)
    doWorkNumpy(train, test, labels)
