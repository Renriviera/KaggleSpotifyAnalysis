from multiprocessing import Pool, freeze_support
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from functools import partial

# function for plotting cross-validation results
def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, kind):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)
    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
    ax.set_title(f"{kind} classifier", fontsize=16)
    ax.set_xlabel('n_estimators', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    # ax.set_ylim(0.2, 0.9)
    ax.set_xticks(depths)
    ax.legend()
    plt.savefig(f"imgs/crossVal{kind}.png")

def run_cross_validation_on_trees_gradient(xTrain, yTrain, n_estimator):
	model = GradientBoostingClassifier(max_depth=3, verbose=0, subsample=1.0, n_estimators=n_estimator) # tune n_estimators=100 and subsample=1.0
	cv_scores = cross_val_score(model, xTrain, yTrain, cv=5, scoring='accuracy')
	print('gradient_boosting, n_estimator:', n_estimator, cv_scores.mean())
	score = model.fit(xTrain, yTrain).score(xTrain, yTrain)
	return [cv_scores.mean(), cv_scores.std(), score]

def run_cross_validation_on_trees_ada(xTrain, yTrain, n_estimator):
	base_estimator = DecisionTreeClassifier(max_depth=n_estimator)
	model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50)
	cv_scores = cross_val_score(model, xTrain, yTrain, cv=5, scoring='accuracy')
	print('ada_boosting, max_depth:', n_estimator, cv_scores.mean())
	score = model.fit(xTrain, yTrain).score(xTrain, yTrain)
	return [cv_scores.mean(), cv_scores.std(), score]

if __name__ == '__main__':
	freeze_support()
	testData = pd.read_csv('archive/train.csv')
	testData.dropna(inplace=True)
	xTrain = testData.iloc[:, 2:16]
	yTrain = testData.iloc[:, 16]
	xTrain, yTrain = shuffle(xTrain, yTrain)
	sm_tree_depths = range(10, 101, 10)
	pool_obj = Pool()
	scores = pool_obj.map(partial(run_cross_validation_on_trees_gradient, xTrain, yTrain), sm_tree_depths)
	scores = np.array(scores).transpose()
	print(scores)
	cv_scores_mean, cv_scores_std, accuracy_scores = scores
	plot_cross_validation_on_trees(sm_tree_depths, cv_scores_mean, cv_scores_std, accuracy_scores, 'gradient_boosting')