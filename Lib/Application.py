import numpy as np
import torch
from .Model import FastRCNN
from .Dataset_Processed import Dataset_Processed, Config_Processed
from .SelectiveSearch import runSelectiveSearch


def visualizeModel(model, dataset, index) -> None:
	model = model.cpu()
	model.train(False)

	# get from dataset
	data = dataset[index]
	image_list		= data[Dataset_Processed.Label.IMAGE_LIST]
	roi_list 		= data[Dataset_Processed.Label.ROI_LIST]
	roi_index_list	= np.zeros((roi_list.shape[0],))

	# to tensor
	image_list		= torch.tensor(image_list,		dtype=torch.float, requires_grad=False)
	roi_list		= torch.tensor(roi_list, 		dtype=torch.float, requires_grad=False)
	roi_index_list	= torch.tensor(roi_index_list, 	dtype=torch.float, requires_grad=False)

	predict_class, predict_box = model(image_list, roi_list, roi_index_list)

	print(predict_class, predict_box)
	print(torch.argmax(predict_class, dim=1))


if __name__ == '__main__':
	# model
	model_ = FastRCNN()

	state_dict = torch.load("./Model/ModelStateDict20201214225501.tar")
	model_.load_state_dict(state_dict)

	# dataset
	config = Config_Processed()
	config.file = "./Data/val/Data.json"
	config.load()

	dataset_ = Dataset_Processed(config, data_path="./Data/val")

	# execute model
	# TODO: not yet completed

	# visualize result
	visualizeModel(model_, dataset_, 0)
