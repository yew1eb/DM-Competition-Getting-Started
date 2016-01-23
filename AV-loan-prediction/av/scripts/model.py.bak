from helper import inverse_mapping

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

import numpy as np

class Model(object):

    def __init__(self):
        pass

    def grid_search(self, X, y, est, parameters, scoring):
        # perform 5 stratified K-Fold
        skf = StratifiedKFold(y, 5)

        fitmodel = GridSearchCV(est, param_grid=parameters, cv=skf, scoring=scoring)
        fitmodel.fit(X, y)

        return fitmodel.best_estimator_, fitmodel.best_params_, fitmodel.best_score_, fitmodel.grid_scores_


"""
Model to generate baseline submission i.e.
model that always predict the most prevalent class
as its output.
"""
class BasicModel(Model):
    def __init__(self, train_df, test_df, target_class):
        super(BasicModel, self).__init__(train_df, test_df, target_class)

    def predict(self):
        most_prominent_class = np.argmax(self.train_df[self.target_class].value_counts())
        baseline_prediction = [most_prominent_class] * self.test_df.shape[0]
        baseline_prediction = map(inverse_mapping, baseline_prediction)

        baseline_prediction = np.array(baseline_prediction)

        return baseline_prediction
