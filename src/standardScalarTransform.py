import pandas as pd
from sklearn.preprocessing import StandardScaler

testData = pd.read_csv('archive/train.csv')
testData.dropna(inplace=True)
yTrain = testData.iloc[:, 16]
xTrain = testData.iloc[:, 2:16]
testDataColumns = testData.columns[2:16]
# print(testDataColumns)
standard_scalar = StandardScaler()
xTrain = pd.DataFrame(standard_scalar.fit_transform(xTrain), columns=testDataColumns).round(3)
standardScalarTrain = pd.concat([testData.iloc[:, 0:2], xTrain, yTrain], axis=1)
print(standardScalarTrain)
standardScalarTrain.to_csv("archive/standardScalarTrain.csv", index=False)