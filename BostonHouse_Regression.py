# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 08:07:32 2020

@author: KIIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

boston = load_boston()

df = pd.DataFrame(boston.data)

df = df.iloc[:,12]

X = np.array(df)

X = preprocessing.scale(X)

Y = np.array(boston.target)

#plot data vs target
plt.scatter(X, Y, color='g')
plt.xlabel('LSTATS')
plt.ylabel('Cost')
plt.show

X = X.reshape(-1, 1)

clf = LinearRegression()
clf.fit(X, Y)

pred = clf.predict(X)

#fitting the regression line
plt.scatter(X, Y, color='g')
plt.plot(X, pred, color='r')
plt.xlabel('LSTATS')
plt.ylabel('Cost')
plt.show

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(3), clf)
model.fit(X,Y)

pred = model.predict(X)

#fitting the regression line
plt.scatter(X, Y, color='g')
plt.plot(X, pred, color='r')
plt.xlabel('LSTATS')
plt.ylabel('Cost')
plt.show


