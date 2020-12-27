from typing import *
import random
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pylab as plt
from .ModelInfo import TrainResultInfo


# plot confusion matrix
# https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
def plotConfusionMatrix(
		info: TrainResultInfo, label_predicted: List[str], label_true: List[str],
		normalize=False, is_show=True) -> None:

	# get confusion matrix
	confusion_matrix = info.confusion_matrix

	# normalize if needed
	if normalize:
		sample_sum 			= np.sum(confusion_matrix)
		confusion_matrix	= confusion_matrix.astype(np.float32)
		confusion_matrix 	/= sample_sum

	# plot graph
	data_frame = pd.DataFrame(confusion_matrix, index=label_true, columns=label_predicted)
	plt.figure(figsize=confusion_matrix.shape)
	sn.heatmap(data_frame, annot=True, fmt=".2f")

	# show if needed
	if is_show:
		plt.show()


# info_list: array structure: [info_index][iteration_index]
# the reason of multiple index is for comparison
# the length of label_info should be the same as info_list
def plotAccuracy(info_list: List[List[TrainResultInfo]], label_info: List[str], is_show=True) -> None:
	line_list: List[Any] = []

	for info in info_list:
		# generate accuracy history for each info
		result_list: List[float] = []
		for iteration in info:
			result_list.append(iteration.getAccuracy())

		# plot graph
		# the reason why need commas
		# https://stackoverflow.com/questions/11983024/matplotlib-legends-not-working
		line, = plt.plot(result_list)
		line_list.append(line)

	# show the result
	plt.ylabel("Accuracy")
	plt.xlabel("Iteration")
	plt.legend(line_list, label_info, loc="upper right")

	# show if needed
	if is_show:
		plt.show()


# info_list: array structure: [info_index][iteration_index]
# the reason of multiple index is for comparison
# the length of label_info should be the same as info_list
def plotLoss(info_list: List[List[TrainResultInfo]], label_info: List[str], is_show=True) -> None:
	line_list: List[Any] = []

	for info in info_list:
		# generate accuracy history for each info
		result_list: List[float] = []
		for iteration in info:
			result_list.append(iteration.loss)

		# plot graph
		plt.plot(result_list)

		# plot graph
		line, = plt.plot(result_list)
		line_list.append(line)

	# show the result
	plt.ylabel("Loss")
	plt.xlabel("Iteration")
	plt.legend(line_list, label_info, loc="upper right")

	# show if needed
	if is_show:
		plt.show()


if __name__ == '__main__':
	# ----- generate info -----
	info_list_: List[List[TrainResultInfo]] = []

	for index_info in range(3):
		info_: List[TrainResultInfo] = []

		# generate result history for each info
		for index_iteration in range(100):
			matrix_ = np.random.randint(0, high=100, size=9)
			matrix_ = matrix_.reshape((3, 3))
			loss_ 	= random.random()

			iteration_ 	= TrainResultInfo(matrix_, loss_)
			info_.append(iteration_)

		# append to list
		info_list_.append(info_)

	# ----- generate info label -----
	label_list_: List[str] = [f"Result_{i}" for i in range(3)]

	# ----- plot -----
	# plot confusion matrix
	plotConfusionMatrix(info_list_[0][0], ["P1", "P2", "P3"], ["T1", "T2", "T3"], normalize=False)
	plotConfusionMatrix(info_list_[0][0], ["P1", "P2", "P3"], ["T1", "T2", "T3"], normalize=True)

	# plot accuracy
	plotAccuracy(info_list_, label_list_)

	# plot loss
	plotLoss(info_list_, label_list_)
