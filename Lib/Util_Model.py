from typing import *
import numpy as np
import torch
from .Util import clipBox_xywh, computeIOU, clipBox


# assume: format: predict_box & y_box: xywh
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

	box_list[:, 0] = box_list[:, 0] + box_list[:, 2] * offset_list[:, 0]
	box_list[:, 1] = box_list[:, 1] + box_list[:, 3] * offset_list[:, 1]
	box_list[:, 2] = box_list[:, 2] * np.exp(offset_list[:, 2])
	box_list[:, 3] = box_list[:, 3] * np.exp(offset_list[:, 3])

	return box_list


# ----- version 1 -----
# below accept tensor instead of np.ndarray
# assume: box_list: format: x1y1x2y2
def getAnchor_topLeft(box_list, size_grid, size_image) -> Any:
	position = (box_list[:, 0:2]).clone()
	box_list = _getAnchor_(position, size_grid, size_image)
	return box_list


def getAnchor_topRight(box_list, size_grid, size_image) -> Any:
	position_x = (box_list[:, 2:3]).clone()
	position_y = (box_list[:, 1:2]).clone()

	position = torch.cat((position_x, position_y), 1)
	box_list = _getAnchor_(position, size_grid, size_image)
	return box_list


def getAnchor_bottomLeft(box_list, size_grid, size_image) -> Any:
	position_x = (box_list[:, 0:1]).clone()
	position_y = (box_list[:, 3:4]).clone()

	position = torch.cat((position_x, position_y), 1)
	box_list = _getAnchor_(position, size_grid, size_image)
	return box_list


def getAnchor_bottomRight(box_list, size_grid, size_image) -> Any:
	position = (box_list[:, 2:4]).clone()
	box_list = _getAnchor_(position, size_grid, size_image)
	return box_list


# ----- version 2 -----
# below accept tensor instead of np.ndarray
# assume: box_list: format: x1y1x2y2
def getAnchor_left(box_list, size_image) -> Any:
	box_w = box_list[:, 2] - box_list[:, 0]

	box_1 = box_list[:, 0] - box_w
	box_2 = box_list[:, 1]
	box_3 = box_list[:, 0]
	box_4 = box_list[:, 3]

	# reshape for torch.cat
	box_1 = box_1.view((-1, 1))
	box_2 = box_2.view((-1, 1))
	box_3 = box_3.view((-1, 1))
	box_4 = box_4.view((-1, 1))

	box_list = torch.cat((box_1, box_2, box_3, box_4), 1)
	box_list = _clipBox_(box_list, size_image)

	return box_list


def getAnchor_right(box_list, size_image) -> Any:
	box_w = box_list[:, 2] - box_list[:, 0]

	box_1 = box_list[:, 2]
	box_2 = box_list[:, 1]
	box_3 = box_list[:, 2] + box_w
	box_4 = box_list[:, 3]

	# reshape for torch.cat
	box_1 = box_1.view((-1, 1))
	box_2 = box_2.view((-1, 1))
	box_3 = box_3.view((-1, 1))
	box_4 = box_4.view((-1, 1))

	box_list = torch.cat((box_1, box_2, box_3, box_4), 1)
	box_list = _clipBox_(box_list, size_image)

	return box_list


def getAnchor_top(box_list, size_image) -> Any:
	box_h = box_list[:, 3] - box_list[:, 1]

	box_1 = box_list[:, 0]
	box_2 = box_list[:, 1] - box_h
	box_3 = box_list[:, 2]
	box_4 = box_list[:, 1]

	# reshape for torch.cat
	box_1 = box_1.view((-1, 1))
	box_2 = box_2.view((-1, 1))
	box_3 = box_3.view((-1, 1))
	box_4 = box_4.view((-1, 1))

	box_list = torch.cat((box_1, box_2, box_3, box_4), 1)
	box_list = _clipBox_(box_list, size_image)

	return box_list


def getAnchor_bottom(box_list, size_image) -> Any:
	box_h = box_list[:, 3] - box_list[:, 1]

	box_1 = box_list[:, 0]
	box_2 = box_list[:, 3]
	box_3 = box_list[:, 2]
	box_4 = box_list[:, 3] + box_h

	# reshape for torch.cat
	box_1 = box_1.view((-1, 1))
	box_2 = box_2.view((-1, 1))
	box_3 = box_3.view((-1, 1))
	box_4 = box_4.view((-1, 1))

	box_list = torch.cat((box_1, box_2, box_3, box_4), 1)
	box_list = _clipBox_(box_list, size_image)

	return box_list


# ----- version 3 -----
# box_list: tensor
def getAnchor(box_list, resize_ratio: Tuple[float, float], size_image: Tuple[int, int]) -> Any:
	box_w = box_list[:, 2] - box_list[:, 0]
	box_h = box_list[:, 3] - box_list[:, 1]

	box_w_new = box_w * resize_ratio[0]
	box_h_new = box_h * resize_ratio[1]

	box_w_new -= box_w
	box_h_new -= box_h

	box_w_new = torch.floor(box_w_new / 2)
	box_h_new = torch.floor(box_h_new / 2)

	# expand / retract rect point
	box_1 = box_list[:, 0] - box_w_new
	box_2 = box_list[:, 1] - box_h_new
	box_3 = box_list[:, 2] + box_w_new
	box_4 = box_list[:, 3] + box_h_new

	# reshape for torch.cat
	box_1 = box_1.view((-1, 1))
	box_2 = box_2.view((-1, 1))
	box_3 = box_3.view((-1, 1))
	box_4 = box_4.view((-1, 1))

	box_list = torch.cat((box_1, box_2, box_3, box_4), 1)
	# box_list = _clipBox_(box_list, size_image)

	return box_list


def _getAnchor_(position, size_grid, size_image) -> Any:
	position = position.clone()

	grid_h_half = size_grid[1] // 2
	grid_w_half = size_grid[0] // 2

	position_top_left		= position.clone()
	position_bottom_right	= position.clone()

	position_top_left[:, 0] 	-= grid_w_half
	position_top_left[:, 1]		-= grid_h_half
	position_bottom_right[:, 0] += grid_w_half
	position_bottom_right[:, 1] += grid_h_half

	box_list = torch.cat((position_top_left, position_bottom_right), 1)
	box_list = _clipBox_(box_list, size_image)

	return box_list


# TODO: may move to Util
# assume: format: x1y1x2y2
def _clipBox_(box_list, size) -> Any:
	box_list = box_list.clone()

	# if the size of image is 840 * 840
	# then valid value of x and y should be both [0, 839]
	box_list[:, 0] = torch.clamp(box_list[:, 0], min=0,	max=size[0] - 1)
	box_list[:, 1] = torch.clamp(box_list[:, 1], min=0,	max=size[1] - 1)
	box_list[:, 2] = torch.clamp(box_list[:, 2], min=0,	max=size[0] - 1)
	box_list[:, 3] = torch.clamp(box_list[:, 3], min=0,	max=size[1] - 1)

	return box_list
