# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 21:26:57 2018
Kmean with bc wisconsin
@author: shifuddin
"""
from load_data import load_zip, load_X_y
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score
import pandas as pd

'''
Load X, y from uri
'''
uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip'
df = load_zip(uri,'Youtube01-Psy.csv', 0, 4, 4, 5, False)
df_dummy = pd.get_dummies(df, drop_first=True)
X, y = load_X_y(df_dummy, 1, 1392, 0, 1)


'''
Fitting K-Means to the dataset
'''
kmeans = KMeans(n_clusters = 10, init = 'k-means++', random_state = 42, max_iter = 1000)
y_kmeans = kmeans.fit_predict(X)
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_
homo_score = homogeneity_score(y.ravel(), y_kmeans)



