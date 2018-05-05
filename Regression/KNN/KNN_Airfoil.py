# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:41:28 2018

@author: shifuddin
"""

from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
'''
Load feature values as X and target as Y
'''
X,y = load_linnerud(return_X_y=True)

'''
Split into training and test set
'''
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.2, random_state=1)

knn_regressor = neighbors.KNeighborsRegressor(algorithm='auto', n_neighbors= 4, weights = 'uniform')
knn_regressor.fit(X_train, y_train)

y_pred = knn_regressor.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, y_pred))

