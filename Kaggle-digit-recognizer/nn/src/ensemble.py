__author__ = 'MarcoGiancarli, m.a.giancarli@gmail.com'


# This program is to be used to combine the results of an ensemble of trained
# networks. It reads the results from the benchmark csv files and write the mode
# into a file named 'ensemble_benchmark.csv'.

from csv import writer
from csv import QUOTE_NONE

BASE_PATH = '../gen/'

files_in_ensemble = [BASE_PATH + name for name in [
    # 'nn_benchmark.csv',
    'nn_benchmark1.csv',
    'nn_benchmark2.csv',
    'nn_benchmark3.csv',
    'nn_benchmark4.csv',
    'nn_benchmark5.csv',
]]

input_files = [open(file_name,'r') for file_name in files_in_ensemble]
for f in input_files:
    f.readline()

with open(BASE_PATH+'ensemble_benchmark.csv', 'wb') as output_file:
    w = writer(output_file, delimiter=',', quoting=QUOTE_NONE)
    w.writerow(['ImageId','Label'])

    for ex_count in range(28000):
        current_predictions = [] # list of digits given by several data files for the same image
        prediction_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # number of times each label appears

        # get the label column for the next line for each data file
        for input_file in input_files:
            current_predictions.append(input_file.readline().split(',')[1])

        # add one to the counter at the index of each label from the data files
        for prediction in current_predictions:
            prediction_counts[int(prediction)] += 1

        # the max index is the label that was predicted most frequently
        averaged_prediction = max(range(len(prediction_counts)),key=prediction_counts.__getitem__)

        print(str(ex_count+1) + ' -- Predictions: ' + ', '.join(current_predictions) + \
                ' -- Average: ' + str(averaged_prediction))
        w.writerow([ex_count+1, averaged_prediction])