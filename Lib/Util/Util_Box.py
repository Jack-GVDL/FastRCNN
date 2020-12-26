from typing import *
import numpy as np
import torch


# Data Structure
# ...


# Function
# def wrap_inplace(func) -> Any:
#
# 	def wrapper(
# 			box_list: Union[np.ndarray, torch.tensor],
# 			is_inplace: bool) \
# 			-> Union[np.ndarray, torch.tensor]:
#
# 		# if not inplace operation
# 		# then copying is needed
# 		if not is_inplace:
# 			if type(box_list) == np.ndarray:
# 				box_list = box_list.copy()
# 			elif type(box_list) == torch.tensor:
# 				box_list = box_list.clone()
# 			else:
# 				raise TypeError
#
# 		return func(box_list)
#
# 	# return wrapped function
# 	return wrapper


def inplace_operation(
		box_list: Union[np.ndarray, torch.Tensor],
		is_inplace: bool = False) -> \
		Union[np.ndarray, torch.Tensor]:

	if is_inplace:
		return box_list

	if type(box_list) == np.ndarray:
		return box_list.copy()
	elif type(box_list) == torch.Tensor:
		return box_list.clone()
	else:
		raise TypeError


def convertBox_xywh_x1y1x2y2(
		box_list: Union[np.ndarray, torch.Tensor],
		is_inplace: bool = False) -> \
		Union[np.ndarray, torch.Tensor]:

	box_list = inplace_operation(box_list, is_inplace)

	box_list[:, 2] += box_list[:, 0]
	box_list[:, 3] += box_list[:, 1]
	return box_list


def convertBox_x1y1x2y2_xywh(
		box_list: 	Union[np.ndarray, torch.Tensor],
		is_inplace: bool = False) -> \
		Union[np.ndarray, torch.Tensor]:

	box_list = inplace_operation(box_list, is_inplace)

	box_list[:, 2] -= box_list[:, 0]
	box_list[:, 3] -= box_list[:, 1]
	return box_list


def getBox_Center_xywh(
		box_list: 	Union[np.ndarray, torch.Tensor],
		is_inplace: bool = False) -> \
		Union[np.ndarray, torch.Tensor]:

	box_list = inplace_operation(box_list, is_inplace)

	box_list[:, 0] = (box_list[:, 0] + box_list[:, 2] / 2)
	box_list[:, 1] = (box_list[:, 1] + box_list[:, 3] / 2)
	return box_list


def getBox_TopLeft_xywh(
		box_list: 	Union[np.ndarray, torch.Tensor],
		is_inplace: bool = False) -> \
		Union[np.ndarray, torch.Tensor]:

	box_list = inplace_operation(box_list, is_inplace)

	box_list[:, 0] = (box_list[:, 0] - box_list[:, 2] / 2)
	box_list[:, 1] = (box_list[:, 1] - box_list[:, 3] / 2)
	return box_list


def scaleBox(
		box_list: 	Union[np.ndarray, torch.Tensor],
		scale: 		Tuple[float, float],
		is_inplace: bool = False) -> \
		Union[np.ndarray, torch.Tensor]:

	box_list = inplace_operation(box_list, is_inplace)

	box_list[:, 0] *= scale[0]
	box_list[:, 1] *= scale[1]
	box_list[:, 2] *= scale[0]
	box_list[:, 3] *= scale[1]
	return box_list


def clipBox_x1y1x2y2(
		box_list:	Union[np.ndarray, torch.Tensor],
		size: 		Tuple[float, float],
		is_inplace:	bool = False) -> \
		Union[np.ndarray, torch.Tensor]:

	box_list = inplace_operation(box_list, is_inplace)

	# numpy
	if type(box_list) == np.ndarray:
		box_list[:, 0] = np.maximum(box_list[:, 0], 0)
		box_list[:, 1] = np.maximum(box_list[:, 1], 0)
		box_list[:, 2] = np.minimum(box_list[:, 2], size[0])
		box_list[:, 3] = np.minimum(box_list[:, 3], size[1])

	# tensor
	else:
		raise TypeError

	return box_list


def clipBox_xywh(
		box_list:	Union[np.ndarray, torch.Tensor],
		size:		Tuple[float, float],
		is_inplace:	bool = False) -> \
		Union[np.ndarray, torch.Tensor]:

	box_list = inplace_operation(box_list, is_inplace)

	box_list = convertBox_xywh_x1y1x2y2(box_list, 		is_inplace=True)
	box_list = clipBox_x1y1x2y2(		box_list, size, is_inplace=True)
	box_list = convertBox_x1y1x2y2_xywh(box_list, 		is_inplace=True)
	return box_list


def computeIOU_x1y1x2y2(
		boxes_1:	Union[np.ndarray, torch.Tensor],
		boxes_2:	Union[np.ndarray, torch.Tensor]) -> \
		Union[np.ndarray, torch.Tensor]:

	# copy array
	boxes_1 		= inplace_operation(boxes_1, False)
	boxes_2 		= inplace_operation(boxes_2, False)
	intersection	= inplace_operation(boxes_1, False)

	# find the intersection
	if type(intersection) == np.ndarray:
		intersection[:, 0] = np.maximum(boxes_1[:, 0], boxes_2[:, 0])
		intersection[:, 1] = np.maximum(boxes_1[:, 1], boxes_2[:, 1])
		intersection[:, 2] = np.minimum(boxes_1[:, 2], boxes_2[:, 2])
		intersection[:, 3] = np.minimum(boxes_1[:, 3], boxes_2[:, 3])
	else:
		raise TypeError

	# find the union area
	# area = A + B - (intersection of A and B)
	def compute_area(boxes):
		# in (x1, y1, x2, y2) format
		dx = boxes[:, 2] - boxes[:, 0]
		dx[dx < 0] = 0
		dy = boxes[:, 3] - boxes[:, 1]
		dy[dy < 0] = 0
		return dx * dy

	a1 = compute_area(boxes_1)
	a2 = compute_area(boxes_2)
	ia = compute_area(intersection)
	assert((a1 + a2 - ia <= 0).sum() == 0)

	return ia / (a1 + a2 - ia)


def computeIOU_xywh(
		boxes_1:	Union[np.ndarray, torch.Tensor],
		boxes_2:	Union[np.ndarray, torch.Tensor]) -> \
		Union[np.ndarray, torch.Tensor]:

	boxes_1 = convertBox_xywh_x1y1x2y2(boxes_1, is_inplace=False)
	boxes_2 = convertBox_xywh_x1y1x2y2(boxes_2, is_inplace=False)
	return computeIOU_x1y1x2y2(boxes_1, boxes_2)


def normalizeImage(
		image_list:	Union[np.ndarray, torch.Tensor],
		is_inplace:	bool = False) -> \
		Union[np.ndarray, torch.Tensor]:

	image_list = inplace_operation(image_list, is_inplace)

	# numpy
	if type(image_list) == np.ndarray:
		image_max 	= np.max(image_list.reshape(image_list.shape[0], -1), axis=1)
		image_min 	= np.min(image_list.reshape(image_list.shape[0], -1), axis=1)
		image_range = image_max - image_min

		image_list 	= image_list - image_min.reshape((	image_list.shape[0], 1, 1))
		image_list	= image_list / image_range.reshape((image_list.shape[0], 1, 1))

	# tensor
	else:
		raise TypeError

	return image_list


# TODO: remove
# def normalizeImage_ranged(image_list, value_min, value_max) -> Any:
# 	image_list = image_list.copy()
#
# 	# TODO: find a better way
# 	for i in range(image_list.shape[0]):
# 		image_list[i] -= value_min
# 		image_list[i] /= (value_max - value_min)


if __name__ == '__main__':
	array_1 = np.random.randint(10, size=(3, 4))
	array_1 = torch.tensor(array_1)
	print(array_1)

	array_2 = convertBox_x1y1x2y2_xywh(array_1, is_inplace=False)
	print(array_1)
	print(array_2)

	array_3 = convertBox_x1y1x2y2_xywh(array_1, is_inplace=True)
	print(array_1)
	print(array_3)
