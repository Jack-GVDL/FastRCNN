from typing import *
from datetime import datetime
import numpy as np
from tqdm import trange
from Lib.Util.Util import computeIOU, clipBox, convert_x1y1x2y2_xywh, convert_xywh_x1y1x2y2
from Lib.Dataset_Processed import Config_Processed

# Function
# result == 0, unclassified
# result == 1, -ve (IOU < negative)
# result == 2, +ve (IOU >= positive), overlap with exactly ONE ground truth box
# result == 3, +ve (IOU >= positive), can overlap with multiple ground truth box
def labelIOU(
		ground_truth, box_list,
		positive: Tuple[float, float] = (0.5, 1.0), negative: Tuple[float, float] = (0.1, 0.5)) -> np.ndarray:

	size_box = box_list.shape[0]
	result_positive = np.zeros((size_box,), dtype=int)
	result_negative = np.zeros((size_box,), dtype=int)

	# foreach ground truth box
	# compute IOU
	for i in range(ground_truth.shape[0]):
		temp = computeIOU(box_list, np.tile(ground_truth[i], (size_box, 1)))

		result_positive += ((temp > positive[0]) & (temp < positive[1])) * 1
		result_negative += ((temp > negative[0]) & (temp < negative[1])) * 1

	return np.zeros((size_box,)) + \
			((result_positive == 0) & (result_negative != 0)) * 1 + \
			(result_positive == 1) * 2 + \
			(result_positive > 1) * 3


def labelDimensionLimit_x1y1x2y2(boxes, size):
	temp_w = boxes[:, 2] - boxes[:, 0]
	temp_h = boxes[:, 3] - boxes[:, 1]

	return (temp_w >= size[0]) & (temp_h >= size[1])


def labelDimensionLimit_xywh(boxes, size):
	temp_w = boxes[:, 2]
	temp_h = boxes[:, 3]

	return (temp_w >= size[0]) & (temp_h >= size[1])


def generateBox(size) -> np.ndarray:
	# get a set of bounding box (in form of x1, y1, x2, y2)
	box_list = np.random.randint(0, high=size, size=(64, 4))

	# fix bounding box based on the rule (x1 < x2, y1 < y2)
	x_min = np.minimum(box_list[:, 0], box_list[:, 2])
	x_max = np.maximum(box_list[:, 0], box_list[:, 2])
	y_min = np.minimum(box_list[:, 1], box_list[:, 3])
	y_max = np.maximum(box_list[:, 1], box_list[:, 3])

	box_list = np.concatenate((
		x_min.reshape((-1, 1)), y_min.reshape((-1, 1)),
		x_max.reshape((-1, 1)), y_max.reshape((-1, 1))),
		axis=1)

	return box_list


def generatePositive(data, size, threshold: Tuple[float, float] = 0.7) -> Dict:
	# parameter
	class_list 	= np.array(data[Config_Processed.Label.CLASS_LIST])
	box_list 	= np.array(data[Config_Processed.Label.BOX_LIST])

	size_box 	= box_list.shape[0]
	roi_per_box = (size + size_box - 1) // size_box  # ceiling

	count = 0
	result_roi 		= np.ndarray((0, 4))
	result_class 	= np.ndarray((0,))
	result_box 		= np.ndarray((0, 4))

	for index_box in range(size_box):

		# for each ground truth bounding box
		count_roi = 0
		while count < size and count_roi < roi_per_box:
			box 			= box_list[index_box]
			box_x1y1x2y2	= convert_xywh_x1y1x2y2(box.copy().reshape((-1, 4)))

			# ----- generate bounding box -----'
			# get bounding box width and height
			box_w = box[2]
			box_h = box[3]

			box_w = box_w // 4
			box_h = box_h // 4

			# generate new roi
			# currently max offset in x and y axis is 50
			offset_list_x1 = np.random.randint(-box_w, high=box_w, size=(roi_per_box, 1))
			offset_list_y1 = np.random.randint(-box_h, high=box_h, size=(roi_per_box, 1))
			offset_list_x2 = np.random.randint(-box_w, high=box_w, size=(roi_per_box, 1))
			offset_list_y2 = np.random.randint(-box_h, high=box_h, size=(roi_per_box, 1))

			offset_list = np.concatenate((offset_list_x1, offset_list_y1, offset_list_x2, offset_list_y2), axis=1)

			roi_list	= np.tile(box_x1y1x2y2, (roi_per_box, 1)) + offset_list
			roi_list	= clipBox(roi_list, (840, 840))
			roi_list	= convert_x1y1x2y2_xywh(roi_list)

			# ----- labeling -----
			# currently use NOT strict rule
			label_iou = labelIOU(box_list, roi_list, positive=threshold)

			# check if width and height larger than a specific size
			# currently the dimension of box should be larger than W=13, H=13
			label_area = labelDimensionLimit_xywh(roi_list, (13, 13))

			# in most of the case
			# as the ground truth bounding box is overlapping
			# strict rule is hard to apply
			label = ((label_iou == 2) | (label_iou == 3)) & (label_area * 1)  # non-strict rule
			# label = (label_iou == 2) & (label_area * 1)  # strict rule

			# ----- append to list -----
			for i in range(label.shape[0]):

				if count >= size:
					break
				if count_roi >= roi_per_box:
					break
				if not label[i]:
					continue

				# add to result
				result_roi = np.concatenate((	result_roi, 	roi_list[i].reshape((1, 4))), 		axis=0)
				result_class = np.concatenate((	result_class, 	class_list[index_box].reshape(1,)), axis=0)
				result_box = np.concatenate((	result_box, 	box.copy().reshape((1, 4))), 		axis=0)

				count_roi 	+= 1
				count 		+= 1

	return {
		Config_Processed.Label.ROI_LIST: 	result_roi,
		Config_Processed.Label.CLASS_LIST: 	result_class,
		Config_Processed.Label.BOX_LIST: 	result_box
	}


def generateNegative(data, size, threshold: Tuple[float, float] = 0.3) -> Dict:
	# parameter
	box_list = np.array(data[Config_Processed.Label.BOX_LIST])

	count	= 0
	result 	= np.ndarray((0, 4))

	# generate negative, bounding box with iou in [0.1, 0.5)
	while count < size:
		# ----- generate bounding box -----
		roi_list = generateBox(840)
		roi_list = convert_x1y1x2y2_xywh(roi_list)

		# ----- labeling -----
		# get / label IOU
		label_iou = labelIOU(box_list, roi_list, negative=threshold)

		# check if width and height larger than a specific size
		# currently the dimension of box should be larger than W=13, H=13
		label_area = labelDimensionLimit_xywh(roi_list, (13, 13))

		# and operation
		label = ((label_iou == 1) * 1) & (label_area * 1)

		# ----- append to list -----
		for i in range(label.shape[0]):

			if count >= size:
				break
			if label[i] == 0:
				continue

			result = np.concatenate((result, roi_list[i].reshape((1, 4))), axis=0)
			count += 1

	return {
		Config_Processed.Label.ROI_LIST: 	result,
		Config_Processed.Label.CLASS_LIST: 	np.zeros((result.shape[0],), dtype=int),
		Config_Processed.Label.BOX_LIST: 	np.ones((result.shape[0], 4), dtype=int)
	}


# generate new ROI with corresponding label
# according to the paper
# Fast R-CNN
#
# 25% of ROI (+ve) should have the IOU with a ground truth bounding box of at least 0.5
# 75% of ROI (-ve) should have the IOU between [0.1, 0.5)
#
# not sure whether foreach +ve ROI
# it should have IOU with an exact ONE ground truth bounding of at least 0.5
# while below 0.5 for rest of the ground truth
def generateROI(
		file_json: str, config_src: Config_Processed,
		positive: Tuple[float, float] = 0.7, negative: Tuple[float, float] = 0.3,
		size_positive: int = 16, size_negative: int = 48) -> None:

	# ----- generate data list -----
	config = Config_Processed()

	# foreach data in dataset
	for i in trange(0, len(config_src)):
		data = config_src[i]

		# get list of positive and negative box
		positive_list = generatePositive(data, size_positive, threshold=positive)
		negative_list = generateNegative(data, size_negative, threshold=negative)

		# merge positive and negative list
		roi_list = np.concatenate((
			positive_list[Config_Processed.Label.ROI_LIST],
			negative_list[Config_Processed.Label.ROI_LIST]),
			axis=0)

		class_list = np.concatenate((
			positive_list[Config_Processed.Label.CLASS_LIST],
			negative_list[Config_Processed.Label.CLASS_LIST]),
			axis=0)

		box_list = np.concatenate((
			positive_list[Config_Processed.Label.BOX_LIST],
			negative_list[Config_Processed.Label.BOX_LIST]),
			axis=0)

		roi_list	= roi_list.astype(np.int32)
		class_list	= class_list.astype(np.int32)
		box_list	= box_list.astype(np.int32)

		# do shuffling
		# how to do shuffling
		# https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order/23289591
		# index_list = np.arange(roi_list.shape[0])
		# np.random.shuffle(index_list)
		#
		# roi_list	= roi_list[index_list]
		# class_list	= class_list[index_list]
		# box_list	= box_list[index_list]

		# add to config
		config.add(
			data[Config_Processed.Label.FILENAME],
			roi_list.tolist(),
			class_list.tolist(),
			box_list.tolist())

	# ----- save config -----
	config.file = file_json
	config.dump()


if __name__ == '__main__':
	# ----- constant -----
	# get time
	# which is used to differentiate different set of ROI generated at different time
	now = datetime.now()
	current_time = now.strftime("%Y%m%d%H%M%S")

	iou_positive = (0.75, 1.0)
	iou_negative = (0.1, 0.75)
	size_positive = 32
	size_negative = 32

	file_src = "../Data/train/DataRaw.json"
	file_src = "../Data/val/Data.json"

	# file_dst = f"../Data/train/Data_" +\
	file_dst = f"../Data/val/Data_" + \
				f"{current_time}_{iou_positive[0]}_{iou_positive[1]}_{iou_negative[0]}_{iou_negative[1]}_" +\
				f"{size_positive}_{size_negative}.json"

	# get dataset
	config_src = Config_Processed()
	config_src.file = file_src
	config_src.load()

	# generate ROI
	generateROI(
		file_dst, config_src,
		positive=iou_positive, negative=iou_negative,
		size_positive=size_positive, size_negative=size_negative)

	print(f"Generate file: {file_dst}")

	# show result
	# config_dst = Config_Processed()
	# config_dst.file = file_dst
	# config_dst.load()
	#
	# dataset = Dataset_Processed(config_dst, data_path="./Data/test")
