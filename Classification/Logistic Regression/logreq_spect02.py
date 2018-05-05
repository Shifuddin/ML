# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 21:42:13 2018

@author: shifuddin
"""

from load_data import load_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

'''
Load X, y from uri
'''
uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train'
X, y = load_csv(uri,',', 1, 24, 0, 1, True)

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
cm = confusion_matrix(y_test.ravel(), y_pred)



