# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 21:26:57 2018
Kmean with bc wisconsin
@author: shifuddin
"""
from load_data import load_csv
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score

'''
Load X, y from uri
'''
uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
X, y = load_csv(uri,',', 1,5, 9,10, True)


'''
Fitting K-Means to the dataset
'''
kmeans = KMeans(n_clusters = 10, init = 'k-means++', random_state = 42, max_iter = 1000)
y_kmeans = kmeans.fit_predict(X)
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_
homo_score = homogeneity_score(y.ravel(), y_kmeans)



