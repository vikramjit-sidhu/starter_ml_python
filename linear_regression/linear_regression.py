
import random
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


dataset_size = 100
data_step_size = 2
data_variance = 50	
data_correlation = 'negative'

X_predict = [8]


def main():
	X, y = get_sample_data(dataset_size, data_variance, data_step_size, correlation=data_correlation)
	m = get_line_slope_for_data(X, y)
	c = get_y_intercept_for_line(X, y, m)
	best_fit_line = get_values_of_line_for_Xs(X, m, c)
	y_predict = predict_points_using_regression_line(X_predict, m, c)
	r_squared = get_coefficient_of_determination(y, best_fit_line)
	print('R squared error is {0}'.format(r_squared))
	plot_points_and_line(X, y, X_predict, y_predict, best_fit_line)
	

def get_sample_data(size_dataset, variance, step=2, correlation=False):
	sample_X = [i for i in range(size_dataset)]
	sample_y = []
	seed_y_value = 1
	for i in range(size_dataset):
		current_y = seed_y_value + random.randrange(-variance, variance)
		sample_y.append(current_y)
		if correlation and correlation == 'positive':
			seed_y_value += step
		elif correlation and correlation == 'negative':
			seed_y_value -= step
	return (np.array(sample_X, dtype=np.float64), np.array(sample_y, dtype=np.float64))
	
	
def get_line_slope_for_data(X, y):
	numerator = mean(X) * mean(y) - mean(X*y)
	denominator = mean(X) ** 2 - mean(X ** 2)
	slope = numerator / denominator
	return slope

	
def get_y_intercept_for_line(X, y, m):
	c = mean(y) - m * mean(X)
	return c
	

def get_values_of_line_for_Xs(X, m, c):
	line = []
	for x in X:
		line.append(m*x + c)
	return line

	
def predict_points_using_regression_line(X_predict, m, c):
	ys = []
	for x in X_predict:
		ys.append(m*x + c)
	return ys
	
	
def plot_points_and_line(X, y, X_predict, y_predict, line):
	style.use('ggplot')
	plt.scatter(X, y, color='r', s=10, label='data')
	plt.scatter(X_predict, y_predict, s=50, color='g', label='prediction')
	plt.plot(X, line, color='b', label='regression line')
	plt.show()


def get_coefficient_of_determination(y_orig, y_pred):
	y_baseline_model = np.repeat(mean(y_orig), len(y_orig))
	sse_baseline = sum_squared_errors(y_orig, y_baseline_model)
	sse_model = sum_squared_errors(y_orig, y_pred)
	r_squared = 1 - (sse_model / sse_baseline)
	return r_squared
	
	
def sum_squared_errors(y_orig, y_pred):
	errors = (y_orig - y_pred) ** 2
	return sum(errors)
	
	
if __name__ == '__main__':
	main()