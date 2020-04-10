# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 03:14:35 2020

@author: KIIT
"""
import numpy as np
import matplotlib.pyplot as plt

X =np.array([(1,2), (3,4), (7,8), (8,9), (3,8), (4,2)])
colors = ['r', 'g']*10

plt.scatter(X[:,0],X[:,1])


class Kmeans:
    def __init__(self,k=2,tol=0.001,max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self,data):
        self.centroids = {}
        
        for i in range(self.k):
            self.centroids[i] = data[i]
            
        for i in range(self.max_iter):
            self.classifications = {}
            
            for i in range(self.k):
                self.classifications[i] = []
                
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid])
                for centroid in self.centroids]
                
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
                
            prev_cent = dict(self.centroids)
            
            optimized = True
            
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
                
            for c in self.centroids:
                orig_cent = prev_cent[c]
                curr_cent = self.centroids[c]
                
            if np.sum((curr_cent-orig_cent)/orig_cent*100) > self.tol:
                optimized = False
                
            if optimized:
                break
            
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
    clf = Kmeans()
    clf.fit(X)
    #clf.predict(X[3])
    
    for centroid in clf.centroids:
        plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                    marker = 'o', color = 'k', s=100, linewidth=5 )
        
    for classification in clf.classifications:
        color = colors[classification]
        for featureset in clf.classifications[classification]:
            plt.scatter(featureset[0], featureset[1], marker="x", color = color, s=100)
            
            
            
   u =np.array([(3,2), (4,4), (5,8), (6,9), (7,8), (8,2)])
   
   for a in u:
       classification = clf.predict(a)
       plt.scatter(a[0], a[1], marker='*',color=colors[classification],s=100)
            
                
            