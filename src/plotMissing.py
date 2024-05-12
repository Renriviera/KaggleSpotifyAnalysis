import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# read data
testData = pd.read_csv('archive/trainWithNaNs.csv')
testData = testData[testData.isnull().any(1)]
xTrain = testData.iloc[:, 2:16]
yTrain = testData.iloc[:, 16]

min_max_scaler = MinMaxScaler()
xTrain = pd.DataFrame(min_max_scaler.fit_transform(xTrain), columns=xTrain.keys().values)
print(xTrain)

# for all features
features = []
i = 0
for feature in xTrain:
  features.append(feature)
  plt.scatter(yTrain - [i for j in range(len(yTrain))], list(xTrain[feature]), s=3)
  i += .05
plt.xticks(range(0, 11))
plt.xlabel('class')
plt.ylabel('scaled value')
plt.legend(features, loc=(1.04,0))
plt.savefig("plots/missing/allFeaturesMissing.png", bbox_inches='tight')
plt.clf()

for feature in xTrain:
  # for individual plots
  plt.scatter(yTrain, list(xTrain[feature]))
  plt.title(feature + " vs. class")
  plt.ylabel('class')
  plt.xlabel(feature)
  plt.savefig("plots/missing/" + feature + ".png")
  plt.clf()