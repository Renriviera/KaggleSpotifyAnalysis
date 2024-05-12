

import sys
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
import sklearn.tree as tree
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

baggingClassifier = False
randomForestClassifier = False
gradientBoostingClassifier = False
useSmote = False

# function for fitting trees of various depths on the training data using cross-validation
def run_cross_validation_on_trees(X, y, tree_depths, cv=5, scoring='accuracy'):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for depth in tree_depths:
        # the BaggingClassifier is actually worse by cross validation than the DecisionTreeClassifier
        if baggingClassifier:
            model = BaggingClassifier(ExtraTreeClassifier(), n_jobs=8, random_state=16, n_estimators=depth, max_features=14)
        elif randomForestClassifier:
            model = RandomForestClassifier(n_estimators=100, n_jobs=8, max_depth=depth)
        elif gradientBoostingClassifier:
            model = GradientBoostingClassifier(max_depth=depth, verbose=1, subsample=.5, n_estimators=100) # tune n_estimators=100 and subsample=1.0
        else:
            model = DecisionTreeClassifier(max_depth=depth)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        print('gradient_boosting depth:' if gradientBoostingClassifier else 'bag n_estimators:' if (baggingClassifier) else 'randomforest depth:' if (randomForestClassifier) else 'depth:', depth, cv_scores.mean())
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        score = model.fit(X, y).score(X, y)
        accuracy_scores.append(score)
        # if type(model) == DecisionTreeClassifier:
        #     fig, ax = plt.subplots(1, 1, figsize=(50, 50))
        #     tree.plot_tree(model, fontsize=2, ax=ax, feature_names=['popularity','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_in_ms','time_signature'])
        #     plt.savefig("imgs/dTree" + str(depth) + ".png", dpi=200)
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores
  
# function for plotting cross-validation results
def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)
    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('n_estimators' if baggingClassifier else 'Random Forest Tree Depth' if randomForestClassifier else 'Tree depth', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    # ax.set_ylim(0.2, 0.9)
    ax.set_xticks(depths)
    ax.legend()
    if baggingClassifier:
        plt.savefig(f"imgs/crossValBagging{'SMOTE' if useSmote else ''}.png")
    elif randomForestClassifier:
        plt.savefig(f"imgs/crossValRandomForest{'SMOTE' if useSmote else ''}.png")
    elif gradientBoostingClassifier:
        plt.savefig(f"imgs/crossValGradientBoosting{'SMOTE' if useSmote else ''}.png")
    else:
        plt.savefig(f"imgs/crossValTree{'SMOTE' if useSmote else ''}.png")

if len(sys.argv) > 1:
    # default DecisionTreeClassifier
    baggingClassifier = sys.argv[1] == 'bag'
    randomForestClassifier = sys.argv[1] == 'randomforest'
    gradientBoostingClassifier = sys.argv[1] == 'gradientboosting'
if len(sys.argv) > 2:
    useSmote = sys.argv[2] == 'smote'
# fitting trees/bags of depth/n_estimators 1 to 24
trainData = pd.read_csv(f"archive/{'smoteT' if useSmote else 't'}rain.csv")
# print(X_train.isnull().sum())
trainData.dropna(inplace=True)
X_train = trainData.iloc[:, 2:16]
y_train = trainData.iloc[:, 16]
X_train, y_train = shuffle(X_train, y_train)
# print(X_train, y_train)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
sm_tree_depths = range(1,40)
sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(X_train, y_train, sm_tree_depths)
# print(sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores)
print(max(sm_cv_scores_mean), list(sm_cv_scores_mean).index(max(sm_cv_scores_mean)) + 1)

# plotting accuracy
title = 'Accuracy per decision tree depth on training data'
if baggingClassifier:
    title = 'Accuracy per bagging classifier n_estimators on training data'
elif randomForestClassifier:
    title = 'Accuracy per random forest tree depth on training data'
plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores, title)