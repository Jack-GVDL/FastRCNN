import os
from typing import *
from .TrainProcess import TrainProcess
from .ModelInfo import ModelInfo


class TrainProcess_Folder(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.name = "Folder"

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	# operation
	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		path 	= info.save_path
		folder 	= info.save_folder

		# check if folder exist or not
		# if not exist, then create one
		folder_path = os.path.join(path, folder)
		if not os.path.isdir(folder_path):
			os.mkdir(folder_path)

	# info
	def getPrintContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	def getLogContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	# Protected
	def _getContent_(self, info: ModelInfo) -> str:
		return "Operation: folder"
