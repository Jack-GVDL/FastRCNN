import os
from typing import *
from .ModelInfo import TrainProcess, ModelInfo, TrainResultInfo


class TrainProcess_Folder(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		# ...

		# operation
		# default stage (can be changed by user)
		self.stage.append(ModelInfo.Stage.TRAIN_START)
		# self.stage.append(ModelInfo.Stage.TRAIN_END)

	def __del__(self):
		return

	# Operation
	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		path 	= info.save_path
		folder 	= info.save_folder

		# check if folder exist or not
		# if not exist, then create one
		folder_path = os.path.join(path, folder)
		if not os.path.isdir(folder_path):
			os.mkdir(folder_path)
