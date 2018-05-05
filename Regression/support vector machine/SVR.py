# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 19:36:21 2018

@author: shifuddin
"""

from DataPreprocessing import DataPreprocessing
preprocessing = DataPreprocessing('Position_Salaries.csv')

features, outcome = preprocessing.read_data(1, 2, 2)

features_scaled, outcome_scaled = preprocessing.scale_data(features, outcome)

from sklearn.svm import SVR
import numpy as np
regressor = SVR(kernel='rbf')
regressor.fit(features_scaled, outcome_scaled)

predicted_salary_scaled = regressor.predict(preprocessing.scale_features(np.array([[6.5]])))

predicted_salary = preprocessing.reverse_outcome(predicted_salary_scaled)

import matplotlib.pyplot as plt
# Visualising the SVR results
plt.scatter(features_scaled, outcome_scaled, color = 'red')
plt.plot(features_scaled, regressor.predict(features_scaled), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
