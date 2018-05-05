# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 00:44:45 2018
kmean with iris dataset
@author: shifuddin
"""
from sklearn.metrics.cluster import homogeneity_score
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

'''
Load X, y from uri
'''
X, y = load_iris(return_X_y=True)

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42, max_iter = 1000)
y_kmeans = kmeans.fit_predict(X)

homo_score = homogeneity_score(y, y_kmeans)

