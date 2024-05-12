import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import sklearn.tree as tree
from sklearn.preprocessing import MinMaxScaler, StandardScaler

max_depth = 8
random_state = 53

# read data
testData = pd.read_csv('archive/train.csv')
testData.dropna(inplace=True)
yTrain = testData.iloc[:, 16]
xTrain = testData.iloc[:, 2:16]
# dTreeLabels = xTrain.keys().values
# this hurts predictions, I'm guessing because it scales time to a smaller scale
# min_max_scaler = MinMaxScaler()
# xTrain = min_max_scaler.fit_transform(xTrain)
# xTest = pd.read_csv("archive/test.csv")

# print('xTrain', xTrain, 'yTrain', yTrain, sep='\n')

# log_reg = LogisticRegression()
# log_reg.fit(xTrain, yTrain)
# prediction = log_reg.predict(xTrain) - yTrain
# print('prediction', prediction, sep='\n')
# score = log_reg.score(xTrain, yTrain)
# print('score', score, sep='\n')

# lin_reg = LinearRegression()
# lin_reg.fit(xTrain, yTrain)
# prediction = lin_reg.predict(xTrain) - yTrain
# print('prediction', prediction, sep='\n')
# score = lin_reg.score(xTrain, yTrain)
# print('score', score, sep='\n')

# dTree = DecisionTreeClassifier(max_depth=max_depth)
# dTree.fit(xTrain, yTrain)
# prediction = dTree.predict(xTrain) - yTrain
# print('dTree prediction', prediction, sep='\n')
# score = dTree.score(xTrain, yTrain)
# print('dTree score', score, sep='\n')

# random_forest = RandomForestClassifier(n_jobs=4, verbose=0, n_estimators=200, max_depth=8, random_state=random_state)
# random_forest.fit(xTrain, yTrain)
# prediction = random_forest.predict(xTrain) - yTrain
# print('random_forest prediction', prediction, sep='\n')
# score = random_forest.score(xTrain, yTrain)
# print('random_forest score', score, sep='\n')

# bagging_classifier = BaggingClassifier(tree.ExtraTreeClassifier(), n_jobs=8, random_state=random_state, n_estimators=11, max_features=14)
# bagging_classifier.fit(xTrain, yTrain)
# predictions = bagging_classifier.predict(xTrain) - yTrain
# print('bagging_classifier prediction', predictions, sep='\n')
# score = bagging_classifier.score(xTrain, yTrain)
# print('bagging_classifier score', score, sep='\n')

# linear_discriminant = LinearDiscriminantAnalysis()
# linear_discriminant.fit(xTrain, yTrain)
# prediction = linear_discriminant.predict(xTrain) - yTrain
# print('linear_discriminant prediction', prediction, sep='\n')
# score = linear_discriminant.score(xTrain, yTrain)
# print('linear_discriminant score', score, sep='\n')

# ada_boost = AdaBoostClassifier()
# ada_boost.fit(xTrain, yTrain)
# prediction = ada_boost.predict(xTrain) - yTrain
# print('ada_boost prediction', prediction, sep='\n')
# score = ada_boost.score(xTrain, yTrain)
# print('ada_boost score', score, sep='\n')

# gradient_boost = GradientBoostingClassifier()
# gradient_boost.fit(xTrain, yTrain)
# prediction = gradient_boost.predict(xTrain) - yTrain
# print('gradient_boost prediction', prediction, sep='\n')
# score = gradient_boost.score(xTrain, yTrain)
# print('gradient_boost score', score, sep='\n')

# linear_svc = LinearSVC()
# linear_svc.fit(xTrain, yTrain)
# prediction = linear_svc.predict(xTrain) - yTrain
# print('linear_svc prediction', prediction, sep='\n')
# score = linear_svc.score(xTrain, yTrain)
# print('linear_svc score', score, sep='\n')
# cv_scores_mean = cross_val_score(linear_svc, xTrain, yTrain, cv=5).mean()
# print('linear_svc cv_scores_mean', cv_scores_mean, sep='\n')

# log_reg_cv = LogisticRegressionCV(max_iter=1000, solver='lbfgs')
# log_reg_cv.fit(xTrain, yTrain)
# prediction = log_reg_cv.predict(xTrain) - yTrain
# print('log_reg_cv prediction', prediction, sep='\n')
# score = log_reg_cv.score(xTrain, yTrain)
# print('log_reg_cv score', score, sep='\n')
# cv_scores_mean = cross_val_score(log_reg_cv, xTrain, yTrain, cv=5).mean()
# print('log_reg_cv cv_scores_mean', cv_scores_mean, sep='\n')