from imblearn.over_sampling import SMOTE
import pandas as pd
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder

# this creates the same number of samples in each class

# read data
testData = pd.read_csv('archive/train.csv')
testData.dropna(inplace=True)
yTrain = testData.iloc[:, 16]
xTrain = testData.iloc[:, 2:16]
# label encode the target variable
y = LabelEncoder().fit_transform(yTrain)
# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(xTrain, y)
# summarize distribution
counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()

# print(pd.DataFrame(testData.iloc[:, 0:2]), X, pd.DataFrame(y))
smoteData = pd.concat([pd.DataFrame(testData.iloc[:, 0:2]), X, pd.DataFrame(y)], axis=1)
print(smoteData)
smoteData.to_csv('archive/smoteTrain.csv', index=False)