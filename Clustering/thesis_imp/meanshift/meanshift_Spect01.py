# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:53:10 2018
meanshift with iris data
@author: shifuddin
"""
from load_data import load_csv
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import homogeneity_score

'''
Load X, y from uri
'''
uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.test'
X, y = load_csv(uri,',', 1, 24, 0, 1, True)


'''
Calculate bandwidth / radius of each cluster centroid from data
'''
bandwidth = estimate_bandwidth(X, quantile=.1,
                               n_samples=100)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

ms.fit(X)

labels = ms.labels_
centroids = ms.cluster_centers_

homogeneity = homogeneity_score(y.ravel(), labels)
