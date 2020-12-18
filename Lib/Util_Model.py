from typing import *
import numpy as np
from .Util import clipBox_xywh, computeIOU


def computeBoxIOU(predict_class, predict_box, y_class, y_box, theta=0.75):
	# clip the box to [0.0, 1.0]
	# there will be error / UB if either xywh is below 0 in computeIOU() function
	# there already has a copying in clipBox_xywh
	predict_box		= clipBox_xywh(predict_box, (1.0, 1.0))
	y_box			= clipBox_xywh(y_box, (1.0, 1.0))

	iou             = computeIOU(predict_box, y_box)
	true_box        = (iou >= theta)

	# TODO: move to other place
	# backup
	# # find out the correct
	# # correct if
	# # class match and for positive, IOU >= theta
	# positive_class	= (y_class != 0)
	# negative_class	= (y_class == 0)
	#
	# correct_positive = np.count_nonzero((positive_class & true_class & true_box))
	# correct_negative = np.count_nonzero((negative_class & true_class))
	#
	# return (correct_positive + correct_negative) / y_class.shape[0], \
	# 		np.count_nonzero(true_class) / y_class.shape[0]

	# it is assumed that the label==0 is the background and no bounding box will present
	true_class		= (predict_class == y_class)
	positive_class 	= (y_class != 0)
	true_box		= (true_class & positive_class & true_box)

	return true_box


def computeConfusionMatrix(predict_class: np.ndarray, y_class: np.ndarray, class_size: int) -> np.ndarray:
	"""
	Compute the confusion matrix
	input should be in shape of [N], where N is the number of sample
	the value should within [0, ... n - 1], where n is the number of class

	:param predict_class:	[N] x-class (from model)
	:param y_class:			[N] y-class (ground truth)
	:param class_size:		size of class: n
	:return:				confusion_matrix[ground_truth][predicted]
	"""
	# it is assumed that the size of predict_class and y_class are the same
	n = class_size

	# confusion matrix is in shape of [n, n]
	# confusion_matrix: List[List[int]] = [[0 for x in range(n)] for y in range(n)]
	confusion_matrix = np.zeros((n, n), dtype=np.int32)

	# TODO: find a way to do the parallel processing
	# foreach sample
	for i in range(y_class.shape[0]):
		row	= y_class[i]
		col = predict_class[i]
		confusion_matrix[row][col] += 1

	return confusion_matrix


def convert_BoxIOU_ConfusionMatrix(box_iou: np.ndarray, predict_class: np.ndarray, y_class: np.ndarray, class_size) -> np.ndarray:
	"""
	# it is assumed that class==0 is background and has no bounding box
	# so class 0 will be ignored
	#
	# the layout of confusion matrix
	# 					Predicted
	#				1T | 1F | 2T | 2F ...
	# Actual	1T
	#			1F
	#			2T
	#			2F
	# ...
	"""
	# box_iou: true: 0; false: 1
	# which means offset is 0 for truth and 1 for false, based on the description in above confusion matrix
	box_iou = (box_iou == 0) * 1

	# it is assumed that the size of predict_class and y_class are the same
	n = class_size

	# confusion matrix is in shape of [(n - 1) * 2, (n - 1) * 2]
	# class u==0 is background and have no bounding box and therefore is ignored
	# confusion_matrix: List[List[int]] = [[0 for x in range(n)] for y in range(n)]
	confusion_matrix = np.zeros(((n - 1) * 2, (n - 1) * 2), dtype=np.int32)

	# this confusion matrix will only focus on sample that participated in IOU check
	# which mean the sample that
	# - not in background class (u == 0), AND
	# - in the correct class (predicted == y)
	participated = (y_class != 0) & (predict_class == y_class)

	# TODO: find a way to do the parallel processing
	# foreach sample
	for i in range(y_class.shape[0]):

		if not participated[i]:
			continue

		row = (y_class[i] - 1) * 2
		col = (y_class[i] - 1) * 2 + box_iou[i] * 1
		confusion_matrix[row][col] += 1

	print(confusion_matrix)
	breakpoint()

	return confusion_matrix


# it is assume that all the box in the parameter is normalized
# and (x, y) in the box_roi and box_y is the center of the bbox
def getBoxOffset(box_roi, box_y) -> Any:
	x = (box_y[:, 0] - box_roi[:, 0]) / box_roi[:, 2]
	y = (box_y[:, 1] - box_roi[:, 1]) / box_roi[:, 3]
	w = np.log(box_y[:, 2] / box_roi[:, 2])
	h = np.log(box_y[:, 3] / box_roi[:, 3])

	x = x.reshape((x.shape[0], 1))
	y = y.reshape((y.shape[0], 1))
	w = w.reshape((w.shape[0], 1))
	h = h.reshape((h.shape[0], 1))

	return np.concatenate((x, y, w, h), axis=1)


# move the box based on the given offset
def offsetBox(box_list, offset_list) -> Any:
	# do not do inplace operation unless it is clearly inform to user
	box_list = box_list.copy()

	box_list[:, 0] = box_list[:, 0] + offset_list[:, 0]
	box_list[:, 1] = box_list[:, 1] + offset_list[:, 1]
	box_list[:, 2] = box_list[:, 2] * np.exp(offset_list[:, 2])
	box_list[:, 3] = box_list[:, 3] * np.exp(offset_list[:, 3])

	return box_list
