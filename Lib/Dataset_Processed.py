from typing import *
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .Util import *
from .Dataset_Image import Dataset_Image


# Data Structure
class Config_Processed:

	# TODO: the variable name (but not the value) should be the (almost) same as that in Dataset_Processed
	#  try to see if have any chance to merge them
	class Label:
		FILENAME	= "Filename"
		ROI_LIST	= "ROIList"
		CLASS_LIST	= "ClassList"
		BOX_LIST	= "BoxList"

	def __init__(self):
		super().__init__()

		# data
		self.data:	List[Dict]	= []
		self.file:	str 		= None  # full path is required

	# Operation
	def load(self) -> bool:
		if self.file == "":
			return False

		# try to open file
		data = None
		with open(self.file, "r") as f:
			data = f.read()

		# check if data is loaded
		if data is None:
			return False

		# load json format data and extract "Data" in the dict
		data		= json.loads(data)
		self.data	= data["Data"]

		return True

	def dump(self) -> bool:
		if self.file == "":
			return False

		# dump json format
		data = {"Data": self.data}
		data = json.dumps(data, indent=4)

		# try to write to the file
		with open(self.file, "w") as f:
			f.write(data)

		return True

	def add(self, filename, roi_list, class_list, box_list) -> bool:
		"""
		:param filename: filename of the npy file (full path is not required)
		:param roi_list: x - list of roi
		:param class_list: y - list of ground truth class
		:param box_list: y - list of ground truth bounding box
		:return: is operation success or not
		"""
		self.data.append({
			self.Label.FILENAME: 	filename,
			self.Label.ROI_LIST:	roi_list.copy(),
			self.Label.CLASS_LIST: 	class_list.copy(),
			self.Label.BOX_LIST:	box_list.copy()
		})
		return False

	# Operator Overloading
	def __getitem__(self, index) -> Dict:
		return self.data[index]

	def __len__(self) -> int:
		return len(self.data)


class Dataset_Processed(Dataset):

	# TODO: dict can use int as key, to boost performance, change str to int
	class Label:
		IMAGE_LIST:		int = 0
		ROI_LIST:		int	= 1
		CLASS_LIST:		int	= 2
		OFFSET_LIST:	int	= 3
		BOX_LIST:		int	= 4

	def __init__(self, config: Config_Processed, data_path: str = ""):
		super().__init__()

		# data
		self._config	= config
		self._data_path	= data_path

		self.size_positive:	int = 16
		self.size_negative:	int = 48

		self._transform = transforms.Compose([
			# transforms.ToTensor(),  # image is not in standard RGB (255, 255, 255) format, cannot use this
			transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5))
		])

		# operation
		# ...

	# Operator Overloading
	def __getitem__(self, index):
		# specification on return item
		# image_list: 	(1, 4, 840, 840), 	float
		# roi_list:		(-1, 4), 			int
		# class_list:	(-1), 				int
		# offset_list:	(-1, 4), 			int
		# load from config
		data = self._config[index]

		# ----- image -----
		# load image
		path_image 	= os.path.join(self._data_path, data[Config_Processed.Label.FILENAME])
		image_list	= np.load(path_image)

		# image processing
		# - clipping: 		there some image that the width is 841 instead of 840
		# - normalization:	range of image intensity is not within [0.0, 1.0]
		# cropping
		# TODO: may consider to use torchvision.transform
		image_list = image_list[:, :840, :840]

		# normalization
		# normalizeImage will get pixel at range of [0.0, 0.1]
		# but the normally it require the range to be [-1.0, 1.0], so torchvision.Normalize() is need
		image_list = normalizeImage(image_list)
		image_list = torch.tensor(image_list, dtype=torch.float)

		# convert to shape of (1, 4, 840, 840) for later concatenation
		image_list = image_list.reshape((1, 4, 840, 840))

		# ----- roi -----
		roi_list	= np.array(data[Config_Processed.Label.ROI_LIST], dtype=np.int32)

		# ----- class -----
		class_list = np.array(data[Config_Processed.Label.CLASS_LIST], dtype=np.int32)

		# ----- offset -----
		box_list	= np.array(data[Config_Processed.Label.BOX_LIST], dtype=np.int32)
		box_list	= scaleBox(box_list.astype(np.float32), (1 / 840, 1 / 840))

		# get and divide positive and negative
		index_positive = np.where(class_list != 0)[0]
		index_negative = np.where(class_list == 0)[0]

		index_positive = np.random.permutation(index_positive)
		index_negative = np.random.permutation(index_negative)

		index_positive = index_positive[:min(index_positive.shape[0], self.size_positive)]
		index_negative = index_negative[:min(index_negative.shape[0], self.size_negative)]

		class_positive 	= class_list[index_positive]
		roi_positive	= roi_list[index_positive]
		box_positive	= box_list[index_positive]

		class_negative	= class_list[index_negative]
		roi_negative	= roi_list[index_negative]
		box_negative	= box_list[index_negative]

		class_list	= np.concatenate((class_positive, class_negative))
		roi_list	= np.concatenate((roi_positive, roi_negative))
		box_list	= np.concatenate((box_positive, box_negative))

		# get normalized roi
		temp_roi = scaleBox(roi_list.astype(np.float32), (1 / 840, 1 / 840))

		# normalize and get offset
		temp_roi 	= getBox_Center_xywh(temp_roi)
		temp_box 	= getBox_Center_xywh(box_list.copy())
		offset_list = getBoxOffset(temp_roi, temp_box)

		# change to dict
		temp: Dict = {
			self.Label.IMAGE_LIST:	image_list,
			self.Label.ROI_LIST:	roi_list,
			self.Label.CLASS_LIST:	class_list,
			self.Label.OFFSET_LIST:	offset_list,
			self.Label.BOX_LIST:	box_list,

			# self.Label.ROI_LIST:	torch.tensor(roi_list,		dtype=torch.float,	requires_grad=False),
			# self.Label.CLASS_LIST:	torch.tensor(class_list,	dtype=torch.float,	requires_grad=False),
			# self.Label.OFFSET_LIST:	torch.tensor(offset_list,	dtype=torch.long,	requires_grad=False),
			# self.Label.BOX_LIST:	torch.tensor(box_list,		dtype=torch.float,	requires_grad=False)
		}

		return temp

	def __len__(self):
		return len(self._config)


# Operation
# from csv to dataset
def generateJsonFromCSV(file_config: str, dataset: Dataset_Image) -> None:
	# ----- generate data_list -----
	config = Config_Processed()

	# add data cluster to the data_list
	# roi from the same image should be clustered together
	# it is assumed that there must have at least one data in the dataset
	filename	= dataset.get(0, False)["Filename"]
	class_list	= []
	box_list	= []

	class_list.append(dataset.get(0, False)["Class"])
	box_list.append(dataset.get(0, False)["Box"])

	for i in range(1, len(dataset)):
		data = dataset.get(i, False)
		name = data["Filename"]

		if name == filename:
			class_list.append(data["Class"])
			box_list.append(data["Box"])
			continue

		# add to data_list
		config.add(filename, box_list.copy(), class_list.copy(), box_list.copy())

		# reset
		filename = name
		class_list.clear()
		box_list.clear()

		# append current
		class_list.append(data["Class"])
		box_list.append(data["Box"])

	# push the final data cluster to the data_list
	config.add(filename, box_list, class_list, box_list)

	# ----- save config -----
	config.file = file_config
	config.dump()


if __name__ == '__main__':
	# constant
	file_config_train: 	str = "./Data/train/DataRaw.json"
	file_config_val:	str = "./Data/val/Data.json"
	file_config_test:	str = "./Data/test/Data.json"

	file_config_list:	List[Any] = [
		[file_config_train, "train"],
		[file_config_val, 	"val"],
		[file_config_test, 	"test"]
	]

	# convert data
	for data in file_config_list:
		dataset_image = Dataset_Image(data[1], data_path="./Data")
		generateJsonFromCSV(data[0], dataset_image)

	# ----- test if conversion is correct or not -----
	# # get config
	# config_train = Config_Processed()
	# config_train.file = file_config_train
	# config_train.load()
	#
	# # get dataset
	# dataset_processed = Dataset_Processed(config_train, data_path="./Data/train")
	# print(dataset_processed[0])
