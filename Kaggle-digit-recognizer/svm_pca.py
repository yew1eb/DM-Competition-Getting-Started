import numpy
from sklearn.decomposition import PCA
from sklearn.svm import SVC

'''
score: 0.98243
'''
COMPONENT_NUM = 35

path = 'd:/dataset/digits/'
print('Read training data...')
with open(path+'train.csv', 'r') as reader:
    reader.readline()
    train_label = []
    train_data = []
    for line in reader.readlines():
        data = list(map(int, line.rstrip().split(',')))
        train_label.append(data[0])
        train_data.append(data[1:])

print(('Loaded ' + str(len(train_label))))

print('Reduction...')
train_label = numpy.array(train_label)
train_data = numpy.array(train_data)
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(train_data)
train_data = pca.transform(train_data)

print('Train SVM...')
svc = SVC()
svc.fit(train_data, train_label)

print('Read testing data...')
with open(path+'test.csv', 'r') as reader:
    reader.readline()
    test_data = []
    for line in reader.readlines():
        pixels = list(map(int, line.rstrip().split(',')))
        test_data.append(pixels)
print(('Loaded ' + str(len(test_data))))

print('Predicting...')
test_data = numpy.array(test_data)
test_data = pca.transform(test_data)
predict = svc.predict(test_data)

print('Saving...')
with open(path+'predict.csv', 'w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in predict:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')
