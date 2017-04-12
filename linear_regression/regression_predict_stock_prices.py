import pandas
import quandl
import math
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression

import datetime
import matplotlib.pyplot as plt
from matplotlib import style

import pickle


# Remove the warning that pandas gives on chained assignment
pandas.options.mode.chained_assignment = None

# Number of seconds in a day
one_day_in_secs = 86400

pickle_file_name = 'google_stocks_linearregression.pickle'


def main():
	df = import_quandl_wikieod_data()
	df = get_meaningful_features_from_wikieod_data(df)
	forecast_col = get_col_name_to_predict()
	# We fill the NaN columns with an outlier value, which will be recognized by the classifier
	df.fillna(-99999, inplace=True)
	# The value we are trying to predict is the future stock price, because of this, we have to shift the columns
	forecast_col_shift_for_label_prediction = math.ceil(0.01 * len(df))
	df = add_label_col_to_df(df, forecast_col, forecast_col_shift_for_label_prediction)
	df.dropna(inplace=True)
	X_train, X_test, y_train, y_test, X_to_predict = get_train_test_predict_set_from_dataframe(df, forecast_col_shift_for_label_prediction)
	# Using Linear Regression with multi-threading
	classifier = LinearRegression(n_jobs=-1)
	# Train classifier
	classifier.fit(X_train, y_train)
	save_classifier_to_pickle(classifier)
	classifier = load_pickled_classifier()
	# Test classifier
	accuracy = classifier.score(X_test, y_test)
	# Predict future values
	forecast_prediction = classifier.predict(X_to_predict)
	df = add_forecast_col_todf(df, forecast_prediction)
	plot_predicted_values_of_classifier(df)
	
	
def import_quandl_wikieod_data():
	df = quandl.get('WIKI/GOOGL')
	return df

	
def get_meaningful_features_from_wikieod_data(df):
	df = df[['Adj. Open', 'Adj. High', 'Adj. Low','Adj. Close', 'Adj. Volume']]
	# The % of change in day's high and low price of the stock (volatility of stock)
	df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100
	# The % change in opening and closing stock price (stability of stock)
	df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
	df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]
	return df


def get_col_name_to_predict():
	return 'Adj. Close'
	

def add_label_col_to_df(df, forecast_col, forecast_col_shift):
	df['label'] = df[forecast_col].shift(-forecast_col_shift)
	return df


def get_train_test_predict_set_from_dataframe(df, forecast_col_shift_for_label_prediction):
	X = np.array(df.drop('label', axis=1))
	# Scaling the features to be between -1 and 1, this speeds up training
	X = preprocessing.scale(X)
	X_future_predict = X[-forecast_col_shift_for_label_prediction:]
	X_model = X[:-forecast_col_shift_for_label_prediction]
	y = np.array(df['label'])
	y_model = y[:-forecast_col_shift_for_label_prediction]
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X_model, y_model, test_size=.2)
	return (X_train, X_test, y_train, y_test, X_future_predict)
	
	
def add_forecast_col_todf(df, forecast_prediction):
	df['Forecast'] = np.nan
	last_date_in_df = df.iloc[-1].name
	last_date_indf_unix = last_date_in_df.timestamp()
	next_date_unix = last_date_indf_unix + one_day_in_secs
	for i in forecast_prediction:
		next_date_fordf = datetime.datetime.fromtimestamp(next_date_unix)
		next_date_unix += one_day_in_secs
		df.loc[next_date_fordf] = [np.nan for _ in range(len(df.columns)-1)]+[i]
	return df

	
def plot_predicted_values_of_classifier(df):
	style.use('ggplot')
	df['Adj. Close'].plot()
	df['Forecast'].plot()
	plt.legend(loc=4)
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.show()
	

def get_last_date_in_df_unix_timestamp(df):
	last_date = df.iloc[-1].name
	last_date_unix_timestamp = last_date.timestamp()
	return last_date_unix_timestamp

	
def save_classifier_to_pickle(classifier):
	with open(pickle_file_name, 'wb') as f:
		pickle.dump(classifier, f)
	

def load_pickled_classifier():
	pickle_in = open(pickle_file_name, 'rb')
	classifier = pickle.load(pickle_in)
	return classifier
	
	
if __name__ == '__main__':
	main()
