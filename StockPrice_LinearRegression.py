# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 01:19:29 2020

@author: KIIT
"""

import pandas as pd 
import quandl, math
import datetime
import time
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from matplotlib import style
quandl.ApiConfig.api_key = 'P-_j_jmWLWP16rpvAwEr'

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. Close', 'Adj. High', 'Adj. Low', 'Adj. Volume']]
df['HL_Perc'] = (df['Adj. High'] - df['Adj. Low'])*100/df['Adj. Low']
df['Perc_chng'] = (df['Adj. Close'] - df['Adj. Open'])*100/df['Adj. Open']

df = df[['Adj. Close', 'HL_Perc', 'Perc_chng', 'Adj. Volume']]

df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))

df['label'] = df['Adj. Close'].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]


df.dropna(inplace=True)
Y = np.array(df['label'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
a = clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

forecast_set = clf.predict(X_lately)
print(X_lately, forecast_set)
print(forecast_set, accuracy, forecast_out)
df['forecast'] = np.nan

last_date = df.iloc[-1]
#last_unix = int(last_date.strftime('%s'))
#last_unix = last_date.timestamp()
#last_unix = last_date.to_datetime()
#last_unix = time.mktime(last_unix.time_tuple())
#pd.to_datetime(last_date.timestamp())
one_day = 86400
next_unix = last_unix + one_day 

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for a in range(len(df.columns) - 1)] + [i]
    
df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=2)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()






















