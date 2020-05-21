# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 22:01:54 2017

@author: Satyajit Pattnaik
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

#Read the train dataset & create a dataframe
df = pd.read_csv('latest_train.csv')

#Features in the problem statement
features = df.columns

#Target Classes i.e. 33 in this case
targetClasses = pd.read_csv('trainLabels.csv')
targetClasses = targetClasses.iloc[:, 1:len(targetClasses.columns)]

#Taking the maximum values for each row, which gives us the value of Classes i.e. y1, y2, y3...etc
targetClasses = targetClasses.idxmax(axis=1)
targetClasses = targetClasses.iloc[0:9999]

#Assigning the predicted class to the dataframe
df['target'] = targetClasses

#Separating the features
x = df.loc[:, features].values

# Separating the target
y = df.loc[:,['target']].values

# Standardizing the features using StandardScaler
x = StandardScaler().fit_transform(x)

#28 components --> 80%
#34 components --> 85%
#44 components --> 90%
#40 components --> 88.5
#PCA Projection based on the number of components
from sklearn.decomposition import PCA
pca = PCA(n_components = 40)
principal_comp = pca.fit_transform(x)
fit = pca.fit(x)
final_df = pd.DataFrame(data = principal_comp)

finalDf = pd.concat([final_df, df[['target']]], axis = 1)

print(pca.explained_variance_ratio_)

#Calculating the Accuracy based on the new number of Components selected
accuracy = sum(pca.explained_variance_ratio_)*100

#Accuracy Percentage based on the PCA components - 82.28% Accuracy using 30 components
print(("Accuracy Percentage: %s") % accuracy)