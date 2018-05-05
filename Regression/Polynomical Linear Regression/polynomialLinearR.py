import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
import numpy
from sklearn import metrics
from sklearn.cross_validation import train_test_split

def read_dataframe(filename):
	
	# read dataset
	dataset_dataframe = pd.read_csv(filename)

	return dataset_dataframe
def retrieve_feature_set(dataset):
	
	# take whole feature set
	# dataset is of dataframe type
	features = dataset.iloc[:,:-1].values
	
	return features
	
def retrieve_outcome(dataset):

	# take whole output set
	# dataset is of dataframe type
	actual_outcome = dataset.iloc[:,-1].values
	
	return actual_outcome

def fit_model(model, feature_train, outcome_train):
	model.fit(feature_train,outcome_train)
	return model

def predict_future(model, features_test):
	
	# predict the result with test data
	predicted_result = model.predict(features_test)
	
	return predicted_result
	
def dataframe_outcome_test_predicted_result(outcome_test, predicted_result):
	# Create panda data frame
	df = pd.DataFrame({'Actual Outcome':outcome_test, 'Predicted Result':predicted_result})
	return df
	
def RMSE(outcome_test, predicted_result):
	rmse = str(numpy.sqrt(metrics.mean_squared_error(outcome_test, predicted_result)))

	return rmse

'''
Read dataset from csv file
'''
dataset_dataframe = read_dataframe('data3.csv')

'''
Retrieve features set and outcome
'''
features = retrieve_feature_set(dataset_dataframe)
outcome = retrieve_outcome(dataset_dataframe)

'''
Create train and test set
'''
features_train, features_test, outcome_train, outcome_test =train_test_split(features, outcome,test_size=0.2, random_state=0)

from sklearn.preprocessing import PolynomialFeatures
ply_reg = PolynomialFeatures(degree = 2)
features_train_poly = ply_reg.fit_transform(features_train)

lr2 = lm.LinearRegression()
lr2.fit(features_train_poly, outcome_train)

predicted_result = lr2.predict(ply_reg.fit_transform(features_test))

# Create panda data frame
dataframe_results = dataframe_outcome_test_predicted_result(outcome_test, predicted_result)

# print the data frame
print (dataframe_results)


# print root mean square error
rmse = RMSE(outcome_test, predicted_result)
print ("Root mean square error "+ rmse)

features_train_size = []
features_train_room = []
for s in features_train:
    features_train_size.append(s[0])
    features_train_room.append(s[1])
    
features_test_size = []
features_test_room = []

for s in features_test:
    features_test_size.append(s[0])
    features_test_room.append(s[1])

plt.scatter(features_train_size, outcome_train, color='g')
plt.show()
plt.scatter(features_train_room, outcome_train, color='r')
plt.show()
