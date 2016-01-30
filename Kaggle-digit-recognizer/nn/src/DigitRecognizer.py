__author__ = 'Marco Giancarli, m.a.giancarli@gmail.com'


import numpy as np
from math import pi as PI
from csv import reader
from csv import writer
from csv import QUOTE_NONE
from skimage import filters
from skimage import measure
from skimage import transform
from skimage import io
from skimage import viewer
from PyNeural.PyNeural import NeuralNetwork


"""
Return a list of all possible translations of the image that don't cut off any
part of the number.
"""
def all_translations(x):
    translations = [x]
    # image = np.array([x])
    # image.resize((28, 28))
    # if sum(image[0, :]) == 0:
    #     image1 = np.vstack([image[1:, :], image[0:1, :]])
    #     translations.append(image1.reshape((1, -1)).tolist()[0])
    # if sum(image[:, 0]) == 0:
    #     image2 = np.hstack([image[:, 1:], image[:, 0:1]])
    #     translations.append(image2.reshape((1, -1)).tolist()[0])
    # if sum(image[-1, :]) == 0:
    #     image3 = np.vstack([image[-1:, :], image[:-1, :]])
    #     translations.append(image3.reshape((1, -1)).tolist()[0])
    # if sum(image[:, -1]) == 0:
    #     image4 = np.hstack([image[:, -1:], image[:, :-1]])
    #     translations.append(image4.reshape((1, -1)).tolist()[0])

    return translations

"""
Return new features, given an initial list x of raw features.
"""
def get_features(x):
    features = []
    image = np.array([x])
    image.resize((28, 28))
    binary_image = filters.threshold_adaptive(image, 9)

    angles = np.linspace(0, 1, 8) * PI
    h, _, _ = transform.hough_line(filters.sobel(binary_image), theta=angles)
    h_sum = [
        [sum(row[start:start+5]) for start in range(0, 75, 5)]
        for row in zip(*h)
    ]
    features.extend(np.array(h_sum).reshape(1, -1).tolist()[0])

    # moments = measure.moments(binary_image)
    # hu_moments = measure.moments_hu(moments)
    # # reshape: -1 as a dimension size makes the dimension implicit
    # features.extend(moments.reshape((1, -1)).tolist()[0])
    # features.extend(hu_moments.reshape((1, -1)).tolist()[0])

    # h_line, _, _ = transform.hough_line(binary_image)
    # features.extend(np.array(h_line).reshape((1, -1)).tolist()[0])

    return features

def main():
    print('Loading training set...')

    training_x_raw = []
    training_y_raw = []
    training_x = []
    training_y = []
    samples = 0
    m = 0

    with open('res/datasets/train.csv', ) as training_file:
        training_data = reader(training_file, delimiter=',')
        skipped_titles = False
        for line in training_data:
            if not skipped_titles:
                skipped_titles = True
                continue
            fields = list(line)
            training_y_raw = fields[0]
            training_x_raw = fields[1:]
            # remove the labels
            training_y.append(int(training_y_raw))

            for features in all_translations([int(v) for v in training_x_raw]):
                training_x.append(features + get_features(features))
                m += 1

            samples += 1
            if any([samples % 1000 == 0,
                    samples % 100 == 0 and samples < 2000,
                    samples % 10 == 0 and samples < 200]):
                print(samples, 'samples loaded.', m, 'generated samples.')
    print('Done.', m, 'total samples.')

    x_array = np.array(training_x)
    # normalize the training set
    training_x = ((x_array - np.average(x_array)) / np.std(x_array)).tolist()

    layer_sizes = [x_array.shape[1], 121, 10]
    alpha = 0.04
    test_size = m / 4 # 4 fold testing

    print('Training set loaded. Samples:', len(training_x))
    print('Training network (layers: ' + \
          ' -> '.join(map(str, layer_sizes)) + ')...')

    network = NeuralNetwork(layer_sizes, alpha)

    network.train(
        training_x[:-test_size],
        training_y[:-test_size],
        test_inputs=training_x[-test_size:],
        test_outputs=training_y[-test_size:],
        epoch_cap=15,
        error_goal=0.00,
        dropconnect_chance=0.05
    )

    print('Network trained.')

    num_correct = 0
    num_tests = 0
    for x, y in zip(training_x[-2000:], training_y[-2000:]):
        prediction = network.predict(x)
        num_tests += 1
        if int(prediction) == y:
            num_correct += 1
    print(str(num_correct), '/', str(num_tests))

    # clear junk
    network.momentum = None
    network.dropconnect_matrices = None
    training_x = None
    training_y = None
    training_data = None
    training_x_raw = None
    training_y_raw = None

    print('Loading test data...')

    test_x_raw = []
    test_x = []
    test_y = []

    output_file_name = 'gen/nn_benchmark5.csv'

    with open(output_file_name, 'wb') as output_file:
        w = writer(output_file, delimiter=',', quoting=QUOTE_NONE)
        w.writerow(['ImageId','Label'])

        with open('res/datasets/test.csv', ) as test_file:
            test_data = reader(test_file, delimiter=',')
            skipped_titles = False
            num_predictions = 0
            for line in test_data:
                if not skipped_titles:
                    skipped_titles = True
                    continue
                fields = list(line)
                test_x_raw = fields
                # remove the damn labels
                features = [int(val) for val in test_x_raw]
                features.extend(get_features(features))
                test_x.append(features)
                num_predictions += 1
                if num_predictions % 100 == 0:
                    x_array = np.array(test_x)
                    # normalize the test set
                    test_x = (
                        (x_array - np.average(x_array)) / np.std(x_array)
                    ).tolist()
                    for i in range(100):
                        w.writerow([num_predictions-99+i,
                                    network.predict(test_x[i])])
                    test_x = []
                    x_array = []


    print('Predicted labels and stored as "' + output_file_name + '".')

if __name__ == '__main__':
    main()