from typing import *
import json
import os
from .Util_Interface import Interface_DictData
from .ModelInfo import TrainProcess, ModelInfo


class TrainProcess_DictSave(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.save_list:	List[Tuple[Interface_DictData, str]] = []

		# operation
		# default stage (can be changed by user)
		self.stage.append(ModelInfo.Stage.TRAIN_END)

	def __del__(self):
		return

	# Operation
	def add(self, obj: Interface_DictData, filename: str) -> bool:
		self.save_list.append((obj, filename))
		return True

	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		# it is assumed that folder and path must be exist
		path 		= info.save_path
		folder 		= info.save_folder
		folder_path = os.path.join(path, folder)

		# foreach object, save its file
		for data_save in self.save_list:

			# convert data from dict to json file
			obj			= data_save[0]
			filename	= data_save[1]

			data_dict	= obj.getDictData()
			data_json	= json.dumps(data_dict, indent=4)

			# save to dst folder with given filename
			with open(os.path.join(folder_path, filename), "w") as f:
				f.write(data_json)

	def getLogContent(self, stage: int, info: Any) -> str:
		return self._getContent_(info)

	def getPrintContent(self, stage: int, info: Any) -> str:
		return self._getContent_(info)

	# Protected
	def _getContent_(self, info: ModelInfo) -> str:
		result: str = ""
		result 		+= "Operation: save dict file\n"
		result		+= "File:\n"

		for data in self.save_list:
			filename	= data[1]
			result 		+= filename + "\n"

		return result