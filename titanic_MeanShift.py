# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 08:23:49 2020

@author: KIIT
"""

import pandas as pd
import numpy as np

df = pd.read_excel(r'C:\Users\KIIT\AppData\Local\Temp\titanic.xls')
df.dtypes

df.convert_objects(convert_numeric=True)

orig_df = pd.DataFrame.copy(df)

df.drop(['name','body','sex'],1,inplace=True)
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
from sklearn.cluster import MeanShift

X = np.array(df.drop(['survived'],1)).astype(float)
X = preprocessing.scale(X)

Y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

n_clusters = len(np.unique(labels))

labels = clf.labels_
    
orig_df['cluster_group'] = np.nan
    
for i in range(len(X)):
    orig_df['cluster_group'].iloc[i] = labels[i]    
    
survival_rates = {}

for i in range(n):
    tempdf = orig_df[ (orig_df['cluster_group']==float(i)) ]
    survival_cluster = tempdf[ (tempdf['survived']==1) ] 
    survival_rates[i] = len(survival_cluster)/len(tempdf)
    
print(survival_rates)
    
    
    
    
    