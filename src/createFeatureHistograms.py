import pandas as pd
import matplotlib.pyplot as plt

trainData = pd.read_csv('archive/trainNoNaNs_durationMs.csv')
for feature in trainData.iloc[:, 2:17]:
  trainData.hist(column=feature)
  plt.savefig(f"histograms/{feature}.png")

fig = plt.figure(figsize = (15,20))
ax = fig.gca()
trainData.hist(ax = ax)
plt.savefig("histograms/allFeatures.png")