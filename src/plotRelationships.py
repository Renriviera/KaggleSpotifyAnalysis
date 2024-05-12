import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

trainData = pd.read_csv('archive/train.csv')
print(trainData)
# cmap = sns.color_palette("magma") #, as_cmap=True, n_colors=17
sns.pairplot(trainData, hue='Class', kind='hist')
plt.savefig("pairPlotRelationships.png")