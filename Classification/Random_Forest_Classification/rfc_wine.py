# -*- coding: utf-8 -*-
"""
Created on Sat May  5 17:32:57 2018

@author: shifuddin
"""

from load_data import load_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


'''
Load X, y from uri
'''
uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
X, y = load_csv(uri,',', 1, 14, 0, 1, True)

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
Fitting random forest classification to the Training set
'''
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


'''
Predicting the Test set results
'''
y_pred = classifier.predict(X_test)

'''
Create confusion matrix
'''
cm = confusion_matrix(y_test, y_pred)