# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 03:40:25 2020

@author: KIIT
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\KIIT\Documents\Python Scripts\iris.csv')

df.replace('Setosa',0,inplace=True)
df.replace('Virginica',2,inplace=True)
df.replace('Versicolor',1,inplace=True)

x = np.array(df.drop(['variety'],1))
x = preprocessing.scale(x)

y = np.array(df['variety'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)

ex = np.array([1,2,3,4])
ex = ex.reshape(1,-1)
pred=clf.predict(ex)
print(pred, accuracy)