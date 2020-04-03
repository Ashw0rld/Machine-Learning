# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:49:21 2020

@author: KIIT
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\KIIT\Documents\Python Scripts\wdbc.data')

df.drop(['ID'], 1, inplace=True)

df.replace('M', 0, inplace=True)
df.replace('B', 1, inplace=True)

x = np.array(df.drop(['Diagnosis'], 1))
x = preprocessing.scale(x)

y = np.array(df['Diagnosis'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

ex = np.array([4,3,2,5,6,7,8,9,1,2,4,3,2,5,6,7,8,9,1,2,4,3,2,5,6,7,8,9,1,2])
ex = ex.reshape(1, -1)
pred = clf.predict(ex)
print(pred)