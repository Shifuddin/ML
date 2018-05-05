# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:51:18 2018
decission tree with banknote
@author: shifuddin
"""

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
'''
Load feature values as X and target as Y
here we read day dataset
'''
X,y = load_boston(return_X_y=True)


'''
Split into training and test set
'''
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.2, random_state=1)

'''
Fit DecisionTreeRegressor with Bike Day data
'''
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

'''
Predicting result
'''
y_pred = regressor.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
