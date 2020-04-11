# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:43:53 2020

@author: KIIT
"""

import pandas as pd
import numpy as np

df = pd.read_excel(r'C:\Users\KIIT\AppData\Local\Temp\titanic.xls')
df.dtypes

df.convert_objects(convert_numeric=True)

df.drop(['name', 'body','sex'],1,inplace=True)
df.fillna(0,inplace=True)
print(df[df.isna().any(axis=1)])

def handle_non_numerical_data(df):
    columns = df.columns.values
    
    for column in columns:
        text_data_vals = {}
        def convert_into_int(val):
            return text_data_vals[val]
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_vals = set(column_contents)
            x=0
            
            for unique in unique_vals:
                if unique not in text_data_vals:
                    text_data_vals[unique] = x
                    x+=1
                    
            df[column] = list(map(convert_into_int, df[column]))
            
    return df

df = handle_non_numerical_data(df)      
                    
from sklearn import preprocessing
from sklearn.cluster import KMeans

X = np.array(df.drop(['survived'],1)).astype(float)
X = preprocessing.scale(X)

Y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct=0

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == Y[i]:
        correct+=1
        
print(correct/len(X))
    













