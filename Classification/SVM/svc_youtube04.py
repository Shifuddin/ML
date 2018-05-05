# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:51:18 2018
decission tree with banknote
@author: shifuddin
"""

from load_data import load_zip, load_X_y
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd

'''
Load X, y from uri
'''
uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip'
df = load_zip(uri,'Youtube04-Eminem.csv', 0, 4, 4, 5, False)
df_dummy = pd.get_dummies(df, drop_first=True)
X, y = load_X_y(df_dummy, 1, 1392, 0, 1)

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
Fitting Decision Tree Classification to the Training set
'''

classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

'''
Predicting the Test set results
'''
y_pred = classifier.predict(X_test)

'''
Create confusion matrix
'''
cm = confusion_matrix(y_test, y_pred)

