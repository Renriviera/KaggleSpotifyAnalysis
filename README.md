Dataset
https://www.kaggle.com/datasets/purumalgi/music-genre-classification?resource=download&select=train.csv
Important notes
train.csv is the file with NaNs filled in. The original, raw data set is trainWithNaNs.csv
xTrain and yTrain.csv contain the X and y data individually. In all cases I've split the data into X and y in the python code instead, though.
You will find some images that don't have code to generate them anymore. I've tested with a bunch of different stuff and I've some of the code to do something different.
Helpful links
MinMax Scalar https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
Standard Scalar https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
SMOTE: https://towardsdatascience.com/how-to-deal-with-imbalanced-data-in-python-f9b71aba53eb
creates a balanced class distribution from imbalanced data
Missing values (removing NaNs) https://www.analyticsvidhya.com/blog/2021/10/handling-missing-value/
Logistic Regression for multi-class classification https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
Sci-kit learn API https://scikit-learn.org/stable/modules/classes.html
Other folders and images
NaNAnalysis contains some txt files to investigate which classes/features had missing values. Not that useful now.
plots contains scatter plots to show the distribution of data for each feature by class. The histograms are probably more useful.
histograms contains the distribution of data for each feature across all classes. Probably wouldn't be too helpful, but could also create histograms for each feature by class.
imgs is the directory with the cross validation plots. The files cross_val.py and parallel_cross_val.py output to this directory.
Interesting points (i.e. Results):
filling the NaNs with mean increased the accuracy
mean: mean() best
median: median() worse
mode: mode() worst
converting the minutes to milliseconds decreased the accuracy
scaling all variables to [0, 1] doesn't help
Data with No NaNs
GradientBoostingClassifier (.66), Random Forest (.60), Decision Tree (.57), and then AdaBoostClassifier (.54) approx. > Bagging Classifier (.53), LogisticRegressionCV with StandardScalar (.503), LogisticRegression (.30), LinearRegression (.12) has the best crossValidation score so far
Data with no NaNs and Standard Scalar
GradientBoostingClassifier (.673), Random Forest (.607), Decision Tree (.57), and then AdaBoostClassifier (.548) approx. > Bagging Classifier (.536)
One Hot Encoding didn't help anything
(SMOTE)[https://machinelearningmastery.com/multi-class-imbalanced-classification/] helped on the order of ~.002
