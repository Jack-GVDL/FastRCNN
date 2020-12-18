from typing import *
import numpy as np
import os
from torch.utils.data import Dataset
from Lib.Util import plotImage


# Data Structure
class Dataset_Image(Dataset):

	def __init__(self, folder, data_path: str = "./Data"):
		self.folder = folder
		self.data_path = data_path
		meta_data_file_name = data_path + f'/bbox_{folder}.csv'

		# read csv
		with open(meta_data_file_name, 'r') as f:
			self.meta_data = [line.split(',') for line in f.read().split('\n')[1:]]

		# get data
		meta_data: Dict = {}

		for index, data in enumerate(self.meta_data):
			meta_data[index] = {
				'file_name': 	f'{data[0]}',
				'size': 		(int(data[1]), int(data[2])),
				'cls': 			2 if ('sev' in data[3]) else 1,
				'box': 			[int(data[4]), int(data[5]), int(data[6]) - int(data[4]), int(data[7]) - int(data[5])]
			}

		self.meta_data = meta_data

	def get(self, index, is_load_image=True):
		# add data to dict
		meta_data = self.meta_data[index]
		temp: Dict = {
			"Class":		meta_data['cls'],
			"Box":			meta_data['box'],
			"Filename":		meta_data["file_name"]
		}

		# if required, then load the image
		if is_load_image:
			path = self.data_path
			path = os.path.join(path, self.folder)
			path = os.path.join(path, meta_data["file_name"])
			temp["ImageList"] = np.load(path)

		return temp

	def __getitem__(self, index):
		return self.get(index)

	def __len__(self):
		return len(self.meta_data)


# Operation
if __name__ == '__main__':
	# ----- load dataset -----
	dataset_train = Dataset_Image("train", data_path="./Data")

	# print info
	data = dataset_train[0]

	print("Bounding box: ", len(dataset_train))
	print("Channel per image: ", data["ImageList"].shape[0])

	# ----- plot graph -----
	data 			= dataset_train[0]
	prev_image 		= data["ImageList"]
	file_name		= data["Filename"]

	sev_box_list 	= []
	mod_box_list 	= []

	for i in range(50):
		data = dataset_train[i]

		image 	= data["ImageList"]
		cls		= data["Class"]
		box		= data["Box"]
		name	= data["Filename"]

		if name == file_name:
			if cls == 2:  # class == SEV
				sev_box_list.append(box)
			else:
				mod_box_list.append(box)
			continue

		# plot image
		plotImage(prev_image, np.array(sev_box_list), np.array(mod_box_list), np.ndarray((0, 4)))

		# plot specific image
		# if file_name == "HS_H08_20180124_1910_33.24_114.62.npy":
		# 	plotImage(prev_image, np.array(sev_box_list), np.array(mod_box_list))

		# print info
		print(f"Image: {file_name}; SEV: {len(sev_box_list)}; MOD: {len(mod_box_list)}")

		# set current image
		prev_image	= image
		file_name	= name
		sev_box_list.clear()
		mod_box_list.clear()

	# ----- check number of image -----
	# count 		= 1
	# file_name 	= dataset_train[0]["Filename"]
	#
	# for i in range(1, len(dataset_train)):
	# 	name = dataset_train[i]["Filename"]
	# 	if name == file_name:
	# 		continue
	#
	# 	count += 1
	# 	file_name = name
	#
	# print(f"Number of image: {count}")
