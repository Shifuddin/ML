# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:51:18 2018
decission tree with banknote
@author: shifuddin
"""

from load_data import load_excel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
'''
Load feature values as X and target as Y
here we read day dataset
'''
uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
X,y = load_excel(uri, 'Sheet1', 0,8, 8,9)


'''
Split into training and test set
'''
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.2, random_state=1)

'''
Fit DecisionTreeRegressor with Bike Day data
'''
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

'''
Predicting result
'''
y_pred = regressor.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
