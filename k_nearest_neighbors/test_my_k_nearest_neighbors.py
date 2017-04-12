import pandas as pd
import random
from k_nearest_neighbors import k_nearest_neighbors


data_filename = 'breast_cancer_wisconsin.txt'

def main():
	df = get_data_in_df()
	df = handle_missing_data(df)
	df = remove_irrelevant_columns(df)
	df = convert_df_data_to_float(df)
	train_set, test_set = get_train_test_data_from_df(df)
	accuracy = test_my_knn(train_set, test_set)
	print('Classifier accuracy is:', accuracy)
	

def get_data_in_df():
	df = pd.read_csv(data_filename)
	return df

	
def handle_missing_data(df):
	# Classifier accuracy is nearly the same, uncommented method is faster
	df.replace('?', -99999, inplace=True)
	# df.replace('?', np.nan, inplace=True)
	# df.dropna(inplace=True)
	return df
	
	
def remove_irrelevant_columns(df):
	df.drop('id', axis=1, inplace=True)
	return df


def convert_df_data_to_float(df):
	df = df.astype(float).values.tolist()
	return df
	
	
def get_train_test_data_from_df(df):
	test_size = .2
	random.shuffle(df)
	test_set_size = int(test_size * len(df))
	train_data = df[:-test_set_size]
	test_data = df[-test_set_size:]
	# We have to create a dictionary with the labels, as this is the format the data needs to be in
	train_set = {2: [], 4: []}
	test_set = {2: [], 4: []}
	for feature in train_data:
		train_set[feature[-1]].append(feature[:-1])
	for feature in test_data:
		test_set[feature[-1]].append(feature[:-1])
	return (train_set, test_set)
	
	
def test_my_knn(train_set, test_set):
	total_predictions = 0
	correct_predictions = 0
	for category, features in test_set.items():
		for feature in features:
			pred_category = k_nearest_neighbors(train_set, feature, k=5)
			if pred_category == category:
				correct_predictions += 1
			total_predictions += 1
	return (correct_predictions / total_predictions)


if __name__ == '__main__':
	main()