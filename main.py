import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting
import seaborn as sns #good visualizing
import os
import warnings

warnings.filterwarnings('ignore')
# print(os.listdir("../input"))
data = pd.read_csv('Wines.csv')
data.columns = data.columns.str.replace(' ','_')
data.info()

#correlation map view
data.corr() 
f, ax = plt.subplots(figsize = (10,10))
sns.heatmap(data.corr(), annot = True, linewidths=.5, fmt = ".2f", ax=ax)
# plt.show()

fig, axes = plt.subplots(11,11, figsize=(50,50))
for i in range(11):
    for j in range(11):
        axes[i, j].scatter(data.iloc[:,i], data.iloc[:,j], c = data.quality)
        axes[i,j].set_xlabel(data.columns[i])
        axes[i,j].set_ylabel(data.columns[j])
        axes[i,j].legend(data.quality)
plt.show()

g = sns.pairplot(data, hue="quality")

#How many wine quality number is realted with how many unique wines
#print(data['quality'].value_counts())
sns.barplot(data['quality'].unique(),data['quality'].value_counts())
plt.xlabel("Quality Rankings")
plt.ylabel("Number of Red Wine")
plt.title("Distribution of Red Wine Quality Ratings")
plt.show()



