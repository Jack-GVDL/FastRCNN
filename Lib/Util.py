from typing import *
import numpy as np
import matplotlib.pyplot as plt


# Data Structure
# copied from mini-project
class AverageMeter:

	def __init__(self):
		# data
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.cnt = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.cnt = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt


# Function
def convert_xywh_x1y1x2y2(boxes):
	boxes[:, 2] += boxes[:, 0]
	boxes[:, 3] += boxes[:, 1]
	return boxes


def convert_x1y1x2y2_xywh(boxes):
	boxes[:, 2] -= boxes[:, 0]
	boxes[:, 3] -= boxes[:, 1]
	return boxes


# this operation will not change the width and height of the bbox
def getCenterBox(box_list) -> Any:
	box_list[:, 0] = (box_list[:, 0] + box_list[:, 2] / 2)
	box_list[:, 1] = (box_list[:, 1] + box_list[:, 3] / 2)
	return box_list


def getTopLeftBox(box_list) -> Any:
	box_list[:, 0] = (box_list[:, 0] - box_list[:, 2] / 2)
	box_list[:, 1] = (box_list[:, 1] - box_list[:, 3] / 2)
	return box_list


def normalizeBox(box_list, size) -> Any:
	box_list = box_list.copy()

	box_list[:, 0] /= size[0]
	box_list[:, 1] /= size[1]
	box_list[:, 2] /= size[0]
	box_list[:, 3] /= size[1]

	return box_list


# the input box should be in x1y1x2y2 format
def clipBox(box_list, size) -> Any:
	box_list = box_list.copy()

	box_list[:, 0] = np.maximum(box_list[:, 0], 0)
	box_list[:, 1] = np.maximum(box_list[:, 1], 0)
	box_list[:, 2] = np.minimum(box_list[:, 2], size[0])
	box_list[:, 3] = np.minimum(box_list[:, 3], size[1])

	return box_list


def clipBox_xywh(box_list, size) -> Any:
	box_list = convert_xywh_x1y1x2y2(box_list.copy())
	box_list = clipBox(box_list, size)
	box_list = convert_x1y1x2y2_xywh(box_list)
	return box_list


# require format of boxes_1 and boxes_2: xywh
def computeIOU(boxes_1, boxes_2):
	boxes_1 = convert_xywh_x1y1x2y2(boxes_1.copy())
	boxes_2 = convert_xywh_x1y1x2y2(boxes_2.copy())

	# find the intersection
	intersection = boxes_1.copy()
	intersection[:, 0] = np.maximum(boxes_1[:, 0], boxes_2[:, 0])
	intersection[:, 1] = np.maximum(boxes_1[:, 1], boxes_2[:, 1])
	intersection[:, 2] = np.minimum(boxes_1[:, 2], boxes_2[:, 2])
	intersection[:, 3] = np.minimum(boxes_1[:, 3], boxes_2[:, 3])

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


def normalizeImage(image_list) -> Any:
	image_list = image_list.copy()

	# TODO: find a better way
	for i in range(image_list.shape[0]):

		value_min = np.min(image_list[i])
		value_max = np.max(image_list[i])

		image_list[i] -= value_min
		image_list[i] /= (value_max - value_min)

	return image_list


def normalizeImage_ranged(image_list, value_min, value_max) -> Any:
	image_list = image_list.copy()

	# TODO: find a better way
	for i in range(image_list.shape[0]):
		image_list[i] -= value_min
		image_list[i] /= (value_max - value_min)


def plotImage(data, sev_box_list, mod_box_list, null_box_list) -> None:
	# normalize
	image_list = normalizeImage(data)

	def draw_box(box, color):
		x, y, w, h = box
		if x == 0:
			x = 1
		if y == 0:
			y = 1

		plt.gca().add_patch(
			plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2, alpha=0.5))

	# for index_channel in range(data.shape[0]):
	for index_channel in range(1):
		plt.imshow(image_list[index_channel], cmap="gray")

		for box in sev_box_list:
			draw_box(box, "red")

		for box in mod_box_list:
			draw_box(box, "blue")

		for box in null_box_list:
			draw_box(box, "green")

		# it is the function actually make image show on screen
		plt.show()
