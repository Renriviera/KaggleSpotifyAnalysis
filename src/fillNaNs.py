import pandas as pd

# cols with no null: [3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15]
testData = pd.read_csv('archive/trainWithNaNs.csv')
for i in range(0, 11):
  classMean = testData['popularity'][testData['Class'] == i].mean().__round__(3)
  print(classMean)
  testData['popularity'][testData['Class'] == i] = testData['popularity'][testData['Class'] == i].fillna(classMean)
for i in range(0, 11):
  classMean = testData['key'][testData['Class'] == i].mean().__round__(3)
  print(classMean)
  testData['key'][testData['Class'] == i] = testData['key'][testData['Class'] == i].fillna(classMean)
for i in range(0, 11):
  classMean = testData['instrumentalness'][testData['Class'] == i].mean().__round__(3)
  print(classMean)
  testData['instrumentalness'][testData['Class'] == i] = testData['instrumentalness'][testData['Class'] == i].fillna(classMean)
testData.to_csv("./archive/train.csv", index=False)