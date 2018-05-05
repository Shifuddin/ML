# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:23:12 2018

@author: shifuddin
"""

from sklearn.neural_network import MLPClassifier
from load_data import load_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

'''
Load X, y from uri
'''
uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
X, y = load_csv(uri,',', 1,5, 9,10, True)


'''
Split into training and test set
'''
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.2, random_state=1)

'''
Feature scaling 
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
Create classifier from MLPClassifier
'''
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
'''
Train the model with X and y
'''
clf.fit(X_train,y_train)

'''
Predicting the Test set results
'''
y_pred = clf.predict(X_test)

'''
Create confusion matrix
'''
cm = confusion_matrix(y_test, y_pred)