# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 22:42:13 2018

@author: shifuddin
"""
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

'''
Load feature values as X and target as Y
'''
X,y = load_breast_cancer(return_X_y=True)

'''
Split into training and test set
'''
X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.2, random_state=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
Perform logistic regression
'''
classifier = LogisticRegression(random_state= 1, solver='liblinear')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

'''
Evaluate the model
'''
cm = confusion_matrix(y_test, y_pred)
