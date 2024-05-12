import pandas as pd
from sklearn.preprocessing import OneHotEncoder

trainData = pd.read_csv('archive/train.csv')
X_train = trainData.iloc[:, 0:2]
uniqueArtists = list(set(X_train['Artist Name']))
uniqueTracks = list(set(X_train['Track Name']))
uniqueArtists.sort()
uniqueTracks.sort()
print(len(uniqueArtists), len(uniqueTracks))
# thing = {}
# for track in X_train['Track Name']:
#   if track not in thing:
#     thing[track] = 0
#   thing[track] += 1
# for k, v in thing.items():
#   if v > 1:
#     print(k)
ohe = OneHotEncoder(categories=[uniqueArtists, uniqueTracks])
ohe.fit(X_train)
conversion = ohe.transform(X_train)
df = pd.DataFrame(conversion.toarray())
print(df)
originalColumns = trainData.iloc[:, 2:17]
print(originalColumns)
concatted = pd.concat([df, originalColumns], axis=1)
print(concatted)
concatted.to_csv("archive/trainWithOneHotEncoder.csv", index=False)
# matching is confirmed correct
# for index, entry in enumerate(conversion.toarray()):
  # print(index, list(entry)) if entry[0] else None
  # print(index, list(entry)) if entry[9149] else None
  # print(len(entry))
# print(len(list(conversion)))
# print(ohe.categories[0], '\n')
# print(ohe.categories[1])