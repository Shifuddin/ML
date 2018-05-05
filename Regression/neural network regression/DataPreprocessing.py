# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 19:08:29 2018

@author: shifuddin
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessing:
    def __init__(self, filename):
        self.filename = filename
        
    def read_csv(self, feature_start_index, feature_end_index, outcome_index):
        # read dataset
        dataframe = pd.read_csv(self.filename)
        
        self.features = dataframe.iloc[:,feature_start_index:feature_end_index].values
        # dataset is of dataframe type
        self.actual_result = dataframe.iloc[:,outcome_index].values
        
        return self.features, self.actual_result
    
    def split_train_test_data(self, test_size=0.25, random_state=0):
        
        features_train, features_test, outcome_train, outcome_test =train_test_split(self.features, self.actual_result,
                                                                                     test_size=test_size, 
                                                                                     random_state=random_state)
        return features_train, features_test, outcome_train, outcome_test
    
    def scale_data(self, features, outcome):
        self.sc_features = StandardScaler()
        self.sc_outcome = StandardScaler()
        
        features = self.sc_features.fit_transform(features)
        outcome = self.sc_outcome.fit_transform(outcome)
        
        return features, outcome
    
    def scale_features(self, value):
        return self.sc_features.transform(value)
    
    def reverse_outcome(self, value):
        return self.sc_outcome.inverse_transform(value)
