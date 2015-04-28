#Main file

#my python files
from utils import *
from laplacian_mean_map import *

#scientific libraries
import numpy as np


#sklearn utils
from sklearn import metrics
from sklearn.dummy import DummyClassifier

#sklearn algorithms
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

data_path = "/Users/giorgio/Google Drive/Laplacian Mean Map/dataset/"

bag_strategy = 'sex'
train_subsample = 20
if train_subsample: print("Subsample only ", train_subsample, " examples")
X_train, pi, y_train, X_test, y_test = preprocess_adult(data_path, train_subsample=train_subsample, bag_strategy=bag_strategy)



##Dummy regressors
print("Dummy classifier, majority class")
model = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
#train validation
print(metrics.accuracy_score(y_train, model.predict(X_train)))
#test
print(metrics.accuracy_score(y_test, model.predict(X_test)))


##Logistic regression
print("Logisic regression L2 regularization with CV")
model = LogisticRegressionCV(Cs=5, penalty='l2').fit(X_train, y_train)
##train validation
C_opt = model.C_
print("best C:", C_opt[0])
print(metrics.accuracy_score(y_train, model.predict(X_train)))
##test
print(metrics.accuracy_score(y_test, model.predict(X_test)))


#Lasso
print("Logisic regression with Lasso with CV")
model = LogisticRegression(C=C_opt, penalty='l1').fit(X_train, y_train)
##train validation
print(metrics.accuracy_score(y_train, model.predict(X_train)))
##test
print(metrics.accuracy_score(y_test, model.predict(X_test)))


#Baselines for LLP
print("LLP Baseline: predict the round() or the proportions on train")
y_pred = np.array((pi[X_train.index.get_level_values('bag')]).round())
print(metrics.accuracy_score(to_one_one(y_train), to_one_one(y_pred)))
print("No test")

#Laplacian Mean Map
print("LLP")
B = compute_bag_means(X_train)
L = laplacian_matrix(B, d)
model = LaplacianMeanMap(L, B).fit(X_train, pi)
#train validation
print(metrics.accuracy_score(to_one_one(y_train), model.predict(X_train)))
#test
print(metrics.accuracy_score(to_one_one(y_test), model.predict(X_test)))

#Laplacian Mean Map with CV
print("LLP with Grid Search")
alphas = np.logspace(-3,3,10)
model = LaplacianMeanMapGridSearch(alphas=alphas)
model = model.fit(X_train, pi)
