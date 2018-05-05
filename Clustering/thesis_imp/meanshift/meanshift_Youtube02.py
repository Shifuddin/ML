# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:53:10 2018
meanshift with iris data
@author: shifuddin
"""
from load_data import load_zip, load_X_y
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import homogeneity_score
import pandas as pd

'''
Load X, y from uri
'''
uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip'
df = load_zip(uri,'Youtube02-KatyPerry.csv', 0, 4, 4, 5, False)
df_dummy = pd.get_dummies(df, drop_first=True)
X, y = load_X_y(df_dummy, 1, 1392, 0, 1)


'''
Calculate bandwidth / radius of each cluster centroid from data
'''
bandwidth = estimate_bandwidth(X, quantile=.1,
                               n_samples=350)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

ms.fit(X)

labels = ms.labels_
centroids = ms.cluster_centers_

homogeneity = homogeneity_score(y.ravel(), labels)
