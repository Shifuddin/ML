# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:06:19 2018

@author: shifuddin
"""

from DataPreprocessing import DataPreprocessing

preprocessing = DataPreprocessing('Position_Salaries.csv')

features, outcome = preprocessing.read_data(1,2,2)


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state= 0)
regressor.fit(features, outcome)

predicted_salary = regressor.predict(3)


'''
Visualize regressor as graph
'''
import numpy as np
import matplotlib.pyplot as plt
X_grid = np.arange(min(features), max(features), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(features, outcome, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

'''
Visualize regressor as Tree 
'''
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())