# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 05:31:46 2020

@author: KIIT
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\KIIT\Documents\Python Scripts\wine.data')

x = np.array(df.drop(['classs'],1))
x = preprocessing.scale(x
                        )
y = np.array(df['classs'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)
print(accuracy)