from typing import *
import numpy as np
import torch
import torch.nn as nn
from Lib.Util.Util import getCenterBox, getTopLeftBox, scaleBox_xywh
from Lib.Util.Util_Model import offsetBox
from .Model.FastRCNN_Alexnet import FastRCNN_Alexnet


def runRegionCapture(
		model: FastRCNN_Alexnet,
		image_list: np.ndarray, roi_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

	"""
	:param model:
	:param image_list:	[size_channel, H, W]
	:param roi_list: 	[size_box, 4]
	:param size_grid:	[grid_w, grid_h]
	:return: 			(mod_box, sev_box)
	"""
	# get image size
	image_h = image_list.shape[1]
	image_w = image_list.shape[2]

	# ----- run model -----
	# to tensor
	tensor_image = torch.tensor(
		image_list.reshape((1, image_list.shape[0], image_list.shape[1], image_list.shape[2])),
		dtype=torch.float)

	tensor_roi			= torch.tensor(roi_list, 						dtype=torch.float, requires_grad=False)
	tensor_roi_index 	= torch.tensor(np.zeros((roi_list.shape[0],)), 	dtype=torch.float, requires_grad=False)

	tensor_image.requires_grad = False

	# run model
	# TODO: currently run in CPU
	predict_class, predict_offset = model(tensor_image, tensor_roi, tensor_roi_index)
	predict_class	= predict_class
	predict_offset	= predict_offset

	# ----- process predicted data -----
	# apply softmax to predict_class
	predict_class = nn.functional.softmax(predict_class, dim=1)

	# to numpy
	predict_class 	= predict_class.detach().numpy()
	predict_offset 	= predict_offset.detach().numpy()

	# then get argmax
	predict_class = np.argmax(predict_class, axis=1)

	# get predicted box
	temp_list = np.zeros((0, 4))

	for index in range(predict_class.shape[0]):
		temp_list = np.concatenate((temp_list, predict_offset[index][predict_class[index]].reshape((1, 4))))

	predict_offset = temp_list

	# get class and offset that u > 0
	box_index 		= np.nonzero(predict_class)
	predict_class 	= predict_class[box_index]
	predict_offset 	= predict_offset[box_index]
	roi_list 		= roi_list[box_index]

	# ----- get MOD, SEV -----
	# based on predicted class and offset
	# discard NIL class and
	# generate list of bounding box [K, 4] of
	# - MOD
	# - SEV
	# compute new roi
	roi_list			= scaleBox_xywh(roi_list.astype(np.float), (1 / image_w, 1 / image_h))  # roi is not normalized, need normalization
	roi_list 		= getCenterBox(roi_list)

	predict_box = offsetBox(roi_list, predict_offset)
	predict_box = getTopLeftBox(predict_box)
	predict_box = scaleBox_xywh(predict_box, (840, 840))

	# split the list into MOD and SEV
	box_mod_index = (predict_class == 1) * 1
	box_sev_index = (predict_class == 2) * 1

	box_mod_index = np.nonzero(box_mod_index)
	box_sev_index = np.nonzero(box_sev_index)

	box_mod = predict_box[box_mod_index]
	box_sev = predict_box[box_sev_index]

	return box_mod, box_sev
