# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:53:10 2018
meanshift with 3D road data
@author: shifuddin
"""
from load_data import load_csv
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import homogeneity_score


'''
Load X, y from uri
'''
uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt'
X, y = load_csv(uri,',', 1, 4, 0, 1, True)


'''
Calculate bandwidth / radius of each cluster centroid from data
'''
bandwidth = estimate_bandwidth(X, quantile=.1,
                               n_samples=10000)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

ms.fit(X)

labels = ms.labels_
centroids = ms.cluster_centers_

homogeneity_score = homogeneity_score(y.ravel(), labels)
