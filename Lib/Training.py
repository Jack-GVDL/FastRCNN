from typing import *

import numpy as np
import torch
from tqdm import tqdm

from .Util import *
from .TrainProcess import *
from .Dataset_Processed import Dataset_Processed


# Function
def trainBatch_(
		info: ModelInfo,
		image_list, roi_list, roi_index_list, 	# x
		class_list, offset_list, 				# y
		is_val=False) -> Dict:

	# forward and get loss
	predict_class, predict_offset 		= info.model(image_list, roi_list, roi_index_list)
	loss_total, loss_class, loss_box 	= info.model.criterion(predict_class, predict_offset, class_list, offset_list)

	# backward
	# if is training
	if not is_val:
		info.optimizer.zero_grad()
		loss_total.backward()
		info.optimizer.step()

	return {
		"PredictClass":		predict_class.data.cpu().numpy(),
		"PredictOffset":	predict_offset.data.cpu().numpy(),
		"LossTotal":		loss_total.data.cpu().item(),
		# "LossClass":		loss_class.cpu(),
		# "LossBox":			loss_box.cpu()
	}


def trainBatch(info: ModelInfo, data_list: List[Any], is_val=False) -> Any:
	# foreach data in the batch
	# the size of each image is (4 * 840 * 840) => (channel, H, W)
	image_list 		= torch.tensor(np.ndarray((0, 4, 840, 840)), dtype=torch.float)
	roi_list 		= np.ndarray((0, 4), 			dtype=float)
	class_list 		= np.ndarray((0,), 				dtype=int)
	offset_list 	= np.ndarray((0, 4), 			dtype=float)
	box_list		= np.ndarray((0, 4),			dtype=float)
	roi_index_list 	= np.ndarray((0,), 				dtype=float)

	for index, data in enumerate(data_list):
		rois = data[Dataset_Processed.Label.ROI_LIST]

		# image_list 		= np.concatenate((image_list, 		data[Dataset_Processed.Label.IMAGE_LIST]),	axis=0)
		image_list		= torch.cat((image_list, 			data[Dataset_Processed.Label.IMAGE_LIST]),	0)
		roi_list 		= np.concatenate((roi_list, 		rois),										axis=0)
		roi_index_list 	= np.concatenate((roi_index_list, 	np.full((rois.shape[0],), index)), 			axis=0)
		class_list 		= np.concatenate((class_list, 		data[Dataset_Processed.Label.CLASS_LIST]), 	axis=0)
		offset_list 	= np.concatenate((offset_list, 		data[Dataset_Processed.Label.OFFSET_LIST]),	axis=0)
		box_list		= np.concatenate((box_list,			data[Dataset_Processed.Label.BOX_LIST]),	axis=0)

	# move data from host to device (GPU / cuda)
	device = info.device_test if is_val else info.device_train

	# tensor_image_list = torch.tensor(image_list, dtype=torch.float, requires_grad=(not is_val)).to(device)
	tensor_image_list = image_list.to(device)
	tensor_image_list.requires_grad = (not is_val)

	tensor_roi_list 		= torch.tensor(roi_list, 		dtype=torch.float, 	requires_grad=False			).to(device)
	tensor_roi_index_list 	= torch.tensor(roi_index_list, 	dtype=torch.float, 	requires_grad=False			).to(device)
	tensor_class_list 		= torch.tensor(class_list, 		dtype=torch.long, 	requires_grad=False			).to(device)
	tensor_offset_list 		= torch.tensor(offset_list, 	dtype=torch.float, 	requires_grad=False			).to(device)

	# actually train the list of image (forward + backward)
	result = trainBatch_(
		info,
		tensor_image_list, tensor_roi_list, tensor_roi_index_list, 	# x
		tensor_class_list, tensor_offset_list,						# y
		is_val=is_val)

	# append the y_class and y_box
	result["ROI"]		= scaleBox(roi_list, (1 / 840, 1 / 840))
	result["YClass"] 	= class_list
	result["YBox"]		= box_list

	return result


def trainEpoch(dataset, info: ModelInfo, is_val=False, is_acquire_result=False) -> Any:
	# parameter
	# batch_size: number of images in each mini-batch
	batch_size = info.batch_size

	result_roi					= np.ndarray((0, 4),	dtype=np.float)
	result_predict_class 		= np.ndarray((0,), 		dtype=np.int32)
	result_predict_offset		= np.ndarray((0, 4),	dtype=np.float)
	result_y_class				= np.ndarray((0,), 		dtype=np.int32)
	result_y_box				= np.ndarray((0, 4),	dtype=np.float)
	result_loss					= 0.0

	# according to the paper
	# first do the permutation over the image
	# then the image selection for mini-batch will be based on the result of this
	size_dataset = len(dataset)
	permute_list = np.random.permutation(np.arange(0, size_dataset))

	# mini-batch training
	# for i in range(0, len(permute_list), N):
	for i in tqdm(range(0, len(permute_list), batch_size), position=0, leave=True):

		# get data
		data_list: List[Any] = []
		for index_data in permute_list[i:i + batch_size]:
			data_list.append(dataset[index_data])

		# train mini-batch
		result = trainBatch(info, data_list, is_val=is_val)

		# append to epoch_result (not batch_result), only when needed
		if is_acquire_result:
			# need to process the raw result
			# - predict_class need argmax
			# - predict_box need to be selected from the result of predict_class
			roi_list				= result["ROI"]
			predict_class_list		= result["PredictClass"]
			predict_offset_list 	= result["PredictOffset"]
			y_class_list			= result["YClass"]
			y_box_list				= result["YBox"]

			# get predicted class
			predict_class_list	= np.argmax(predict_class_list, axis=1)

			# get predicted box
			temp_list = np.zeros((0, 4))

			for index in range(predict_class_list.shape[0]):
				temp_list = np.concatenate((temp_list, predict_offset_list[index][predict_class_list[index]].reshape(1, 4)))

			# append to result
			result_roi				= np.concatenate((result_roi,				roi_list			))
			result_predict_class 	= np.concatenate((result_predict_class, 	predict_class_list	))
			result_predict_offset	= np.concatenate((result_predict_offset, 	temp_list			))
			result_y_class			= np.concatenate((result_y_class, 			y_class_list		))
			result_y_box			= np.concatenate((result_y_box, 			y_box_list			))
			result_loss				+= result["LossTotal"]

	# return result (only when required)
	result_epoch = {}
	if is_acquire_result:
		result_epoch = {
			"ROI":				result_roi,
			"PredictClass":		result_predict_class,
			"PredictOffset":	result_predict_offset,
			"YClass":			result_y_class,
			"YBox":				result_y_box,
			"LossTotal":		result_loss
		}

	return result_epoch


def train_(dataset: Dataset_Processed, info: ModelInfo) -> Any:
	# ----- train start -----
	info.model.train(True)

	# signal
	info.executeProcess(ModelInfo.Stage.ITERATION_TRAIN_START, {})

	# ----- train -----
	result = trainEpoch(dataset, info, is_val=False, is_acquire_result=True)
	if info.scheduler is not None:
		info.scheduler.step()

	# get data returned from training
	roi_list 		= result["ROI"]  # in xywh(top-left) format
	predict_offset	= result["PredictOffset"]
	predict_class	= result["PredictClass"]
	y_class			= result["YClass"]
	y_box			= result["YBox"]  # in xywh(top-left) format

	# process data returned from training
	roi_list		= getBox_Center_xywh(roi_list)
	predict_box		= offsetBox(roi_list, predict_offset)
	predict_box		= getBox_TopLeft_xywh(predict_box)  # resultant predict_box is in xywh(top-left) format
	loss_cur 		= result["LossTotal"] / (len(dataset) / info.batch_size)

	# get model performance
	confusion_matrix_class	= computeConfusionMatrix(predict_class, y_class, 3)
	box_iou					= computeBoxIOU(predict_class, predict_box, y_class, y_box)
	confusion_matrix_box	= computeConfusionMatrix_BoxIOU(box_iou, predict_class, y_class, 3)
	confusion_matrix_total	= computeConfusionMatrix_Total(box_iou, predict_class, y_class, 3)

	# save the result
	result_info_class		= TrainResultInfo(confusion_matrix_class, 	loss_cur)
	result_info_box			= TrainResultInfo(confusion_matrix_box,		loss_cur)
	result_info_total		= TrainResultInfo(confusion_matrix_total,	loss_cur)

	info.result_list[-1].append(result_info_class)
	info.result_list[-1].append(result_info_box)
	info.result_list[-1].append(result_info_total)

	# ----- train end -----
	# signal
	info.executeProcess(ModelInfo.Stage.ITERATION_TRAIN_END, {})


def validate_(dataset: Dataset_Processed, info: ModelInfo) -> Any:
	# ----- validate start -----
	info.model.train(False)

	# signal
	info.executeProcess(ModelInfo.Stage.ITERATION_VAL_START, {})

	# ----- validate -----
	result = trainEpoch(dataset, info, is_val=True, is_acquire_result=True)

	# get data returned from training
	roi_list 		= result["ROI"]  # in xywh(top-left) format
	predict_offset	= result["PredictOffset"]
	predict_class	= result["PredictClass"]
	y_class			= result["YClass"]
	y_box			= result["YBox"]  # in xywh(top-left) format

	# process data returned from training
	roi_list		= getBox_Center_xywh(roi_list)
	predict_box		= offsetBox(roi_list, predict_offset)
	predict_box		= getBox_TopLeft_xywh(predict_box)  # resultant predict_box is in xywh(top-left) format
	loss_cur 		= result["LossTotal"] / (len(dataset) / info.batch_size)

	# get model performance
	confusion_matrix_class	= computeConfusionMatrix(predict_class, y_class, 3)
	box_iou					= computeBoxIOU(predict_class, predict_box, y_class, y_box)
	confusion_matrix_box	= computeConfusionMatrix_BoxIOU(box_iou, predict_class, y_class, 3)
	confusion_matrix_total	= computeConfusionMatrix_Total(box_iou, predict_class, y_class, 3)

	# save the result
	result_info_class		= TrainResultInfo(confusion_matrix_class, 	loss_cur)
	result_info_box			= TrainResultInfo(confusion_matrix_box,		loss_cur)
	result_info_total		= TrainResultInfo(confusion_matrix_total,	loss_cur)

	info.result_list[-1].append(result_info_class)
	info.result_list[-1].append(result_info_box)
	info.result_list[-1].append(result_info_total)

	# ----- validate end -----
	# signal
	info.executeProcess(ModelInfo.Stage.ITERATION_VAL_END, {})


def train(dataset_list: Dict, info: ModelInfo) -> Any:
	# parameter
	epoch = info.epoch

	# put model to correct place
	info.model = info.model.to(info.device_train)

	# signal
	info.executeProcess(ModelInfo.Stage.TRAIN_START, {})

	# for i in tqdm(range(epoch), position=0, leave=True):
	for i in range(epoch):
		# result
		info.result_list.append([])

		# train
		train_(dataset_list["Train"], info)

		# val
		validate_(dataset_list["Val"], info)

		# ----- iteration end -----
		info.iteration += 1

	# signal
	info.executeProcess(ModelInfo.Stage.TRAIN_END, {})


# Operation
if __name__ == '__main__':
	pass
