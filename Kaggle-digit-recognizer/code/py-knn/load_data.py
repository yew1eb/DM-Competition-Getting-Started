import csv
import numpy as np


# loading csv data into numpy array
def read_data(f, header=True, test=False):
    data = []
    labels = []

    csv_reader = csv.reader(open(f, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index = index + 1
        if header and index == 1:
            continue

        if not test:
            labels.append(int(row[0]))
            row = row[1:]

        data.append(np.array(np.int64(row)))
    return (data, labels)


if __name__ == "__main__":
    train, labels = read_data("../data/train.csv")
    test, tmpl = read_data("../data/test.csv", test=True)
