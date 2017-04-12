import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style
import warnings
from collections import Counter


dataset = {'r':[[6,5], [7,7], [8,6]], 'k':[[1,2], [2,3], [3,1]]}
prediction = [5,7]

def main():
	category = k_nearest_neighbors(dataset, prediction, k=3)
	print(category)
	plot_dataset_and_prediction_pt(dataset, prediction)


def k_nearest_neighbors(dataset, prediction_feature, k=3):
	categories = get_categories_of_dataset(dataset)
	check_num_categories_with_k(categories, k)
	distance_and_category =  find_euclidian_distance_of_prediction_pt_from_all_feature_points(dataset, prediction_feature)
	category = find_category_of_prediction_feature(distance_and_category, k)
	return category
	
	
def get_categories_of_dataset(dataset):
	categories = []
	for category, _ in dataset.items():
		categories.append(category)
	return categories


def check_num_categories_with_k(categories, k):
	if len(categories) <= k:
		return
	warnings.warn('K is set to a value less than the total categories present')
	
	
def find_euclidian_distance_of_prediction_pt_from_all_feature_points(dataset, prediction_feature):
	# List of lists with distance of feature from prediction feature and category the feature belongs to
	distance_and_category = []
	for category, features in dataset.items():
		for feature in features:
			distance = find_euclidean_distance(feature, prediction_feature)
			distance_and_category.append([distance, category])
	return distance_and_category

	
def find_euclidean_distance(p, q):
	# euclidean_distance = np.sqrt(np.sum((p-q) ** 2))
	p = np.array(p)
	q = np.array(q)
	euclidean_distance = np.linalg.norm(p-q)
	return euclidean_distance
	

def find_category_of_prediction_feature(category_and_distance, k):
	k_closest_categories = [category for _,category in sorted(category_and_distance)[:k]]
	category_freq_tuple = Counter(k_closest_categories).most_common(1)
	return category_freq_tuple[0][0]
	
	
def plot_dataset_and_prediction_pt(dataset, prediction):
	# Plots only 2-d features
	style.use('fivethirtyeight')
	for category, features in dataset.items():
		for feature in features:
			plot.scatter(feature[0], feature[1], color=category, s=50)
	plot.scatter(prediction[0], prediction[1], s=50)
	plot.show()
	
	
if __name__ == '__main__':
	main()