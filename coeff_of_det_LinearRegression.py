# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 01:09:32 2020

@author: KIIT
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(hm,var,step=2,correlation='false'):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-var, var)
        ys.append(y)
        if correlation and correlation=='pos':
            val+=step
        elif correlation and correlation=='false':
            val-=step
            
    xs = [i for i in range(len(ys))]
    
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
   

def best_slope_line(xs,ys) :
    m = ( ( (mean(xs)*mean(ys)) - mean(xs*ys) ) / 
          ((mean(xs)**2) - (mean(xs**2))) ) 
    b = mean(ys) - m*mean(xs)
    
    return m, b

def squared_err(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coeff_of_det(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys]
    squar_error_reg = squared_err(ys_orig, ys_line)
    squar_error_mean = squared_err(ys_orig, y_mean_line)
    return 1 - (squar_error_reg/squar_error_mean)

xs, ys = create_dataset(40,10,2,correlation='pos')

m, b = best_slope_line(xs,ys)

print(m,b)

regression_line = [(m*x) + b  for x in xs]

pred_x = 8
pred_y = m*pred_x + b

r = coeff_of_det(ys, regression_line)
print(r)

plt.scatter(xs, ys)
plt.scatter(pred_x, pred_y, color='g')
plt.plot(xs, regression_line)
plt.show()