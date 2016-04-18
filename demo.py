"""
@author: Giorgio Patrini <giorgio.patrini@anu.edu.au>
"""

from nolabelnopy.utils.utils import *
from nolabelnopy.algorithms.laplacian_mean_map import *

import numpy as np

from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


def to_zero_one(label):
    """
    Binary label transformation {-1,+1} -> {0,1}
    """
    return ((label + 1)/2).astype(np.int)


def to_one_one(label):
    """
    binary label transformation {-1,+1} -> {0,1}
    """
    return (2*(label) - 1).astype(np.int)


data_path = "/Users/giorgio/Google Drive/Laplacian Mean Map/dataset/"

bag_strategy = 'race'
train_subsample = 2000
if train_subsample:
    print("Subsample only ", train_subsample, " examples")
X_train, pi, y_train, X_test, y_test = \
    preprocess_adult(data_path, train_subsample=train_subsample,
                     bag_strategy=bag_strategy)

# Dummy regressors
print("Dummy classifier, majority class")
model = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
print("Train %.2f" % metrics.accuracy_score(y_train, model.predict(X_train)))
print("Test %.2f" % metrics.accuracy_score(y_test, model.predict(X_test)))

# Logistic regression
print("Logistic regression L2 regularization with CV")
model = LogisticRegressionCV(Cs=5, penalty='l2').fit(X_train, y_train)
C_opt = model.C_
print("best C:", C_opt[0])
print("Train %.2f" % metrics.accuracy_score(y_train, model.predict(X_train)))
print("Test %.2f" % metrics.accuracy_score(y_test, model.predict(X_test)))

# Lasso
print("Logistic regression with Lasso with CV")
model = LogisticRegression(C=C_opt[0], penalty='l1').fit(X_train, y_train)
print("Train %.2f" % metrics.accuracy_score(y_train, model.predict(X_train)))
print("Test %.2f" % metrics.accuracy_score(y_test, model.predict(X_test)))

# Baselines for LLP
print("LLP Baseline: predict the round() or the proportions on train")
y_pred = np.array((pi[X_train.index.get_level_values('bag')]).round())
print("Train %.2f" % metrics.accuracy_score(to_one_one(y_train), to_one_one(y_pred)))
print("No test")

# Laplacian Mean Map
print("LLP")
B = compute_bag_means(X_train)
L = laplacian_matrix(B, d)
model = LaplacianMeanMap(L, B).fit(X_train, pi)
print("Train %.2f" % metrics.accuracy_score(to_one_one(y_train), model.predict(X_train)))
print("Test %.2f" % metrics.accuracy_score(to_one_one(y_test), model.predict(X_test)))

# Laplacian Mean Map with CV
print("LLP with Grid Search")
alphas = np.logspace(-3, 3, 5)
gammas = np.logspace(-3, 3, 5)
sigmas = np.logspace(-3, -3, 5)
model = LaplacianMeanMapGridSearch(alphas=alphas, gammas=gammas,
                                   sigmas=sigmas).fit(X_train, pi)
print("Train %.2f" % metrics.accuracy_score(to_one_one(y_train), model.predict(X_train)))
print("Test %.2f" % metrics.accuracy_score(to_one_one(y_test), model.predict(X_test)))
