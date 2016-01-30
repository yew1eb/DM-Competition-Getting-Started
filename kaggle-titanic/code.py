import numpy as np
import pandas as pd

from sklearn import cross_validation
from sklearn.cross_validation import KFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

titanic = pd.read_csv("./data/train.csv", dtype={"Age": np.float64}, )

# Preprocessing Data
# ==================

# Fill in missing value in "Age".
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# Replace all the occurences of male with the number 0.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# Convert the Embarked Column.
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2


# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Linear Regression
# =================
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

# Evaluating error and accuracy
predictions = np.concatenate(predictions,axis = 0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

print(('Accuracy of Linear Regression on the training set is ' + str(accuracy)))

# Logistic Regression
# ===================
alg = LogisticRegression()
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(('Accuracy of Logistic Regression on the training set is ' + str(scores.mean())))

# Random Forest
# ===================
from sklearn.ensemble import RandomForestClassifier

alg = RandomForestClassifier(n_estimators=1000,min_samples_leaf=5, max_features="auto", n_jobs=2, random_state=10)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(('Accuracy of Random Forest on the training set is ' + str(scores.mean())))

# Test Set
# ========
titanic_test = pd.read_csv("./data/test.csv", dtype={"Age": np.float64}, )

titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv('result_rf.csv', index=False)