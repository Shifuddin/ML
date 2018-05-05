# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:53:10 2018
meanshift with iris data
@author: shifuddin
"""
from sklearn.datasets import load_iris
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import homogeneity_score

'''
Load X, y from uri
'''
X, y = load_iris(return_X_y=True)


'''
Calculate bandwidth / radius of each cluster centroid from data
'''
bandwidth = estimate_bandwidth(X, quantile=.1,
                               n_samples=1700)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

ms.fit(X)

labels = ms.labels_
centroids = ms.cluster_centers_

homogeneity = homogeneity_score(y.ravel(), labels)
