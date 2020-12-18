from typing import *
import numpy as np
import cmath
import math


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
	# get image and grid dimension
	image_w 		= image.shape[2]
	image_h 		= image.shape[1]
	size_channel 	= image.shape[0]

	grid_w = image_w // size_grid[0]
	grid_h = image_h // size_grid[1]

	# image padding
	# TODO: not yet completed

	# histogram
	# image_h + 1 and image_w + 1 is used for padding
	histogram = np.zeros((image_h + 1, image_w + 1, size_channel * size_bucket))

	label = np.arange(0, size_grid[0] * size_grid[1])
	label = label.reshape((size_grid[1], size_grid[0]))

	# load histogram
	step = np.arange(0, size_channel * size_bucket, size_bucket)

	for y in range(grid_h):
		for x in range(grid_w):

			# find out the mean color in each grid (for each channel)
			color = (image[:, y:y + grid_h, x:x + grid_w]).reshape((size_channel, -1))
			color = np.mean(color, axis=1)

			# put each color intensity into corresponding bucket
			# size_bucket * 0.9999 is to make that the max resultant number is size_bucket - 1
			# e.g. size_bucket = 10
			# then the max resultant value = 1.0 * (10 * 0.9999) = 9
			color = (color * (size_bucket * 0.9999)).astype(np.int32)
			color += step

			# add to histogram
			histogram[y][x][color] = 1

	# compute similarity
	# connectivity is 4
	similarity = np.zeros((grid_h, grid_w, 2))


if __name__ == '__main__':
	image_ = np.random.rand(4, 840, 840)
	runSelectiveSearch(image_, 25, (10, 10))
