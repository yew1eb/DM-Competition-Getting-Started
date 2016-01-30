from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

CPU = 1


def main():
    print("Reading training set")
    dataset = genfromtxt(open('../data/train.csv', 'r'), delimiter=',', dtype='int64')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    print("Reading test set")
    test = genfromtxt(open('../data/test.csv', 'r'), delimiter=',', dtype='int64')[1:]

    #create and train the random forest
    rf = RandomForestClassifier(n_estimators=1000, n_jobs=CPU)
    print("Fitting RF classifier")
    rf.fit(train, target)

    print("Predicting test set")
    savetxt('submission-version-1.csv', rf.predict(test), delimiter=',', fmt='%d')

if __name__ == "__main__":
    main()
