import numpy as np
import pandas as pd
from sklearn import model_selection, neighbors


data_filename = 'breast_cancer_wisconsin.txt'


def main():
	df = get_data_in_df()
	df = handle_missing_data(df)
	df = remove_irrelevant_columns(df)
	X_train, X_test, y_train, y_test = get_train_test_data_from_df(df)
	classifier = get_k_neighbors_classifier()
	classifier.fit(X_train, y_train)
	accuracy = classifier.score(X_test, y_test)
	print('Classifier accuracy is {0}'.format(accuracy))
	

def get_data_in_df():
	df = pd.read_csv(data_filename)
	return df

	
def handle_missing_data(df):
	# Classifier accuracy is nearly the same, this method is faster
	df.replace('?', -99999, inplace=True)
	# df.replace('?', np.nan, inplace=True)
	# df.dropna(inplace=True)
	return df
	
	
def remove_irrelevant_columns(df):
	df.drop('id', axis=1, inplace=True)
	return df

	
def get_train_test_data_from_df(df):
	X = np.array(df.drop('class', axis=1))
	y = np.array(df['class'])
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
	return (X_train, X_test, y_train, y_test)

	
def get_k_neighbors_classifier():
	return neighbors.KNeighborsClassifier()

	
if __name__ == '__main__':
	main()