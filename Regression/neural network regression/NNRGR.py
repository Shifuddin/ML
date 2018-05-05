# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:23:12 2018

@author: shifuddin
"""

from sklearn.neural_network import MLPRegressor
from DataPreprocessing import DataPreprocessing

'''
Create object of preprocessing class
'''
preprocessing = DataPreprocessing('Position_Salaries.csv')

'''
Read features and outcome from csv file
'''
X, Y = preprocessing.read_csv(1,2,2)


'''
transforms and fit features and outcome
'''
X, Y = preprocessing.scale_data(X, Y)

'''
Create classifier from MLPClassifier
'''
regressor = MLPRegressor(hidden_layer_sizes=(100,50))
'''
Train the model with X and y
'''
regressor.fit(X,Y)

'''
Scale the feautre before prediction
Inverse transform the result 
Predict result
'''
preprocessing.reverse_outcome(regressor.predict(preprocessing.scale_features([[10]])))