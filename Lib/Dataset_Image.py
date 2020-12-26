from typing import *
import numpy as np
import os
from torch.utils.data import Dataset


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