import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
import numpy
from sklearn import metrics
from sklearn.cross_validation import train_test_split

def read_dataframe(filename):
	
	# read dataset
	dataset_dataframe = pandas.read_csv(filename)

	return dataset_dataframe
def retrieve_feature_set(dataset):
	
	# take whole feature set
	# dataset is of dataframe type
	features = dataset.iloc[:,:-1].values
	
	return features
	
def retrieve_outcome(dataset):

	# take whole output set
	# dataset is of dataframe type
	actual_outcome = dataset_dataframe.iloc[:,-1].values
	
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
	df = pandas.DataFrame({'Actual Outcome':outcome_test, 'Predicted Result':predicted_result})
	return df
	
def RMSE(outcome_test, predicted_result):
	rmse = str(numpy.sqrt(metrics.mean_squared_error(outcome_test, predicted_result)))

	return rmse
	

def ploting_data(features, results):
	plt.scatter(features, results,color='g', label='data')
	plt.show()

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

# create linear regression model
lr = lm.LinearRegression()

# fit the model with features and outcome
fitted_model = fit_model(lr, features_train, outcome_train)

# Y intercept of the line
print ("Y-intercept of line: " + str(fitted_model.intercept_))

# slope of the line
print ("Coefficients of the line :" + str(fitted_model.coef_))

# predict the result with training data
predicted_result = predict_future(fitted_model, features_test)

# Create panda data frame
dataframe_results = dataframe_outcome_test_predicted_result(outcome_test, predicted_result)

# print the data frame
print (dataframe_results)

# print root mean square error
rmse = RMSE(outcome_test, predicted_result)
print ("Root mean square error "+ rmse)

#plot the model
#ploting_data(features_train, outcome_train)
'''
plt.plot(features_train, lr.predict(features_train), color='k')
plt.title('Size vs Price')
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()
'''


