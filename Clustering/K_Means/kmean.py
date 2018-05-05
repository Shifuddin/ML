# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 00:44:45 2018

@author: shifuddin
"""

from DataPreprocessing import DataPreprocessing

preprocessing = DataPreprocessing('Mall_Customers.csv')

features, outcome = preprocessing.read_data(2,4,4)

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(features)


plt.scatter(features[:,1], features[:,0],c=y_kmeans)
plt.xlabel("Anual Income")
plt.ylabel("Age")