import time
from typing import *
import numpy as np
import cmath
import math
from Lib.Dataset_Processed import Dataset_Processed, Config_Processed
from Lib.Util import normalizeImage, plotImage, convert_x1y1x2y2_xywh, normalizeBox, computeIOU


# Function
def runSelectiveSearch(image: np.ndarray, size_bucket: int, size_grid: Tuple[int, int]) -> np.ndarray:
	"""
	Selective search
	it is assumed that the intensity of image is normalized, i.e. intensity: [0.0, 1.0]

	:param image:		[C, h, w], C: channel of image (can be something other than RGB and can be number other than 3)
	:param size_bucket:	[x]
						where x is the bucket size for each channel,
						then the total number of bucket in the histogram should be x * C
	:param size_grid:	[w, h], which should be the same, but in case of so special condition
	:return:			[K, 4], K: number of bounding box, and bounding box is in format of xywh
	"""
	# result
	# which should be a list of bounding box
	result = np.zeros((0, 4), dtype=int)

	# ----- dimension -----
	# get image and grid dimension
	image_w 		= image.shape[2]
	image_h 		= image.shape[1]
	size_channel 	= image.shape[0]

	grid_w = image_w // size_grid[0]
	grid_h = image_h // size_grid[1]

	# ----- image padding -----
	# image padding
	# TODO: not yet completed

	# ----- histogram -----
	# histogram
	histogram = np.zeros((size_grid[1], size_grid[0], size_channel * size_bucket))

	# load histogram
	step = np.arange(0, size_channel * size_bucket, size_bucket)

	# ----- bounding box -----
	# format of box
	# box[row][col][bounding_box], where bounding box is in form: [x_min, y_min, x_max, y_max]
	box_x = np.tile(np.arange(0, size_grid[0]), 							(size_grid[1], 1)).reshape(
		(size_grid[1], size_grid[0], 1))

	box_y = np.tile(np.arange(0, size_grid[1]).reshape((size_grid[1], 1)), 	(1, size_grid[0])).reshape(
		(size_grid[1], size_grid[0], 1))

	box = np.concatenate((box_x, box_y), 	axis=2)
	box = np.concatenate((box, box), 		axis=2)

	# ----- label -----
	label = np.arange(0, size_grid[0] * size_grid[1])
	label = label.reshape((size_grid[1], size_grid[0]))

	# pad -1 to the end of row and col
	# label = np.concatenate((label, np.ones((1, size_grid[0])) * -1), 		axis=0)
	# label = np.concatenate((label, np.ones((size_grid[1] + 1, 1)) * -1), 	axis=1)

	# change type of label
	label = label.astype(np.int32)

	# TODO: find a better way
	for y in range(size_grid[1]):
		for x in range(size_grid[0]):

			# find out the mean color in each grid (for each channel)
			temp_y = y * grid_h
			temp_x = x * grid_w

			color = (image[:, temp_y:temp_y + grid_h, temp_x:temp_x + grid_w]).reshape((size_channel, -1))
			color = np.mean(color, axis=1)

			# put each color intensity into corresponding bucket
			# size_bucket * 0.9999 is to make that the max resultant number is size_bucket - 1
			# e.g. size_bucket = 10
			# then the max resultant value = 1.0 * (10 * 0.9999) = 9
			color = (color * (size_bucket * 0.9999)).astype(np.int32)
			color += step

			# add to histogram
			histogram[y][x][color] = 1

	# total number of iteration that a big fucking box is formed is size_grid[0] * size_grid[1] - 1
	# you can refer to the number of branch in a binary tree that have leaf node = size_grid[0] * size_grid[1]
	for _ in range(size_grid[0] * size_grid[1] - 1):

		# ----- compute similarity -----
		# compute connectivity mask
		# it is computed as there should be not similarity check between box with same label index
		connect_mask_x = label[:-1, :-1] - label[:-1, 1:]
		connect_mask_y = label[:-1, :-1] - label[1:, :-1]

		connect_mask_x_zero	= (connect_mask_x != 0) * 1
		connect_mask_y_zero	= (connect_mask_y != 0) * 1
		connect_mask_x_one 	= (connect_mask_x == 0) * 1
		connect_mask_y_one 	= (connect_mask_x == 0) * 1

		# get the normalized histogram
		histogram_sum 			= np.sum(histogram, axis=2)
		histogram_sum_mask		= (histogram_sum == 0) * 1
		histogram_sum			+= histogram_sum_mask

		histogram_normalized 	= histogram / histogram_sum.reshape((histogram.shape[0], histogram.shape[1], 1))

		# connectivity is 4 and connection check start from top-left, left-to-right and top-to-down
		# therefore for each box, it should handle right and down box
		similarity_x = np.minimum(histogram_normalized[:-1, :-1], histogram_normalized[:-1, 1:])
		similarity_y = np.minimum(histogram_normalized[:-1, :-1], histogram_normalized[1:, :-1])

		# compute sum in every histogram
		similarity_x = np.sum(similarity_x, axis=2)
		similarity_y = np.sum(similarity_y, axis=2)

		# similarity between the same region is ignored
		similarity_x = similarity_x * connect_mask_x_zero - connect_mask_x_one
		similarity_y = similarity_y * connect_mask_y_zero - connect_mask_y_one

		similarity = np.concatenate((
			similarity_x.reshape(	1, similarity_x.shape[0], similarity_x.shape[1]),
			similarity_y.reshape(	1, similarity_y.shape[0], similarity_y.shape[1])	))

		# ----- find the merge boxes -----
		# merge_1 and merge_2 are in format of xy
		merge_position = np.unravel_index(np.argmax(similarity), similarity.shape)

		merge_1 = np.zeros((2,), dtype=int)
		merge_2 = np.zeros((2,), dtype=int)

		# horizontal
		if merge_position[0] == 0:
			merge_1[0] = merge_position[2]
			merge_2[0] = merge_position[2] + 1
			merge_1[1] = merge_position[1]
			merge_2[1] = merge_position[1]

		# vertical
		else:
			merge_1[0] = merge_position[2]
			merge_2[0] = merge_position[2]
			merge_1[1] = merge_position[1]
			merge_2[1] = merge_position[1] + 1

		# ----- get mask -----
		# where the mask will be used for in union histogram and union label
		label_1 = label[merge_1[1]][merge_1[0]]
		label_2 = label[merge_2[1]][merge_2[0]]

		label_mask_1 = (label == label_1) * 1
		label_mask_2 = (label == label_2) * 1

		# ----- union histogram -----
		# expand label_mask to (size_grid[1] + 1, size_grid[0] + 1)
		# where the shape of histogram is (size_grid[1] + 1, size_grid[0] + 1, _)
		# backup
		# label_mask_1 = np.concatenate((label_mask_1, np.zeros((1, 10))), axis=0)
		# label_mask_1 = np.concatenate((label_mask_1, np.zeros((11, 1))), axis=1)
		# label_mask_2 = np.concatenate((label_mask_2, np.zeros((1, 10))), axis=0)
		# label_mask_2 = np.concatenate((label_mask_2, np.zeros((11, 1))), axis=1)

		# union histogram
		increment_1 = histogram[merge_2[1]][merge_2[0]]
		increment_2 = histogram[merge_1[1]][merge_1[0]]

		increment_1 = np.tile(increment_1, (histogram.shape[0], histogram.shape[1], 1))
		increment_2 = np.tile(increment_2, (histogram.shape[0], histogram.shape[1], 1))

		increment_1 = increment_1 * label_mask_1.reshape((label_mask_1.shape[0], label_mask_1.shape[1], 1))
		increment_2 = increment_2 * label_mask_2.reshape((label_mask_2.shape[0], label_mask_2.shape[1], 1))

		histogram += increment_1
		histogram += increment_2

		# ----- add to result -----
		box_1 = np.unravel_index(label_1, (size_grid[1], size_grid[0]))
		box_2 = np.unravel_index(label_2, (size_grid[1], size_grid[0]))

		box[box_1][0] = min(box[box_1][0], box[box_2][0])
		box[box_1][1] = min(box[box_1][1], box[box_2][1])
		box[box_1][2] = max(box[box_1][2], box[box_2][2])
		box[box_1][3] = max(box[box_1][3], box[box_2][3])

		# box need to add to the result
		box_add = box[box_1]
		box_add = box_add.reshape((1, 4))

		result = np.concatenate((result, box_add))

		# ----- union label -----
		label_dis	= label_mask_2 * (label_1 - label_2)
		label		+= label_dis

	return result


if __name__ == '__main__':
	config_val = Config_Processed()
	config_val.file = "../Data/val/Data.json"
	config_val.load()

	dataset_val = Dataset_Processed(config_val, data_path="../Data/val")

	for i in range(len(dataset_val)):
		# ----- get data -----
		data 	= dataset_val[i]
		image_ 	= data[Dataset_Processed.Label.IMAGE_LIST][0].numpy()
		roi		= data[Dataset_Processed.Label.ROI_LIST]
		box		= data[Dataset_Processed.Label.BOX_LIST]

		# normalize image
		# image_ = normalizeImage(image_)

		grid_size = (10, 10)

		# ----- selective search -----
		start_time = time.time()
		result_ = runSelectiveSearch(image_, 25, grid_size)
		print(f"Runtime: {time.time() - start_time}")

		# ----- plot image -----
		grid_w = 840 // grid_size[0]
		grid_h = 840 // grid_size[1]

		result_ *= np.array([[grid_w, grid_h, grid_w, grid_h]])
		result_plot = result_.copy()

		result_plot[:, 0] += 5
		result_plot[:, 1] += 5
		result_plot[:, 2] -= 5
		result_plot[:, 3] -= 5

		result_plot += np.array([[0, 0, grid_w, grid_h]])
		result_plot = convert_x1y1x2y2_xywh(result_plot)

		plotImage(image_, np.zeros((0, 4)), roi, result_plot)

		# ----- check IOU -----
		result_ = np.unique(result_, axis=0)
		result_ = convert_x1y1x2y2_xywh(result_)
		result_	= result_.astype(np.float)
		result_ = normalizeBox(result_, (840, 840))

		for i in range(box.shape[0]):
			iou = computeIOU(result_, np.tile(box[i], (result_.shape[0], 1)))

			print(np.nonzero(iou > 0.1))
			print(iou[np.nonzero(iou > 0.1)])
			breakpoint()
