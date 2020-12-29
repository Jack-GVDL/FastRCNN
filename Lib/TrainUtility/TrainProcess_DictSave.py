from typing import *
import json
import os
from .Util_Interface import Interface_DictData
from .TrainProcess import TrainProcess
from .ModelInfo import ModelInfo


class TrainProcess_DictSave(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.name = "DictSave"

		self._save_list: List[Tuple[Interface_DictData, str]] = []

		# operation
		# ...

	def __del__(self):
		return

	# Property
	@property
	def save_list(self):
		return self._save_list.copy()

	# Operation
	# data
	def setData(self, data: Dict) -> None:
		self._save_list = self._getDataFromDict_(data, "save_list", self._save_list)

	def getData(self) -> Dict:
		return {
			"save_list": self._save_list
		}

	# operation
	def addDictData(self, obj: Interface_DictData, filename: str) -> bool:
		self._save_list.append((obj, filename))
		return True

	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		# it is assumed that folder and path must be exist
		path 		= info.save_path
		folder 		= info.save_folder
		folder_path = os.path.join(path, folder)

		# foreach object, save its file
		for data_save in self._save_list:

			# convert data from dict to json file
			obj			= data_save[0]
			filename	= data_save[1]

			data_dict	= obj.getDictData()
			data_json	= json.dumps(data_dict, indent=4)

			# save to dst folder with given filename
			with open(os.path.join(folder_path, filename), "w") as f:
				f.write(data_json)

	# info
	def getLogContent(self, stage: int, info: Any) -> str:
		return self._getContent_(info)

	def getPrintContent(self, stage: int, info: Any) -> str:
		return self._getContent_(info)

	def getInfo(self) -> List[List[str]]:
		info: List[List[str]] = []

		# ----- save list -----
		save_list = map(lambda x: ["", x[1]], self._save_list)
		save_list = list(save_list)

		# if the save_list is not empty
		# then the first item will be assigned with a parameter name (save_list)
		if save_list:
			save_list[0][0] = "save_list"

		info.extend(save_list)

		return info

	# Protected
	def _getContent_(self, info: ModelInfo) -> str:
		result: str = ""
		result 		+= "Operation: save dict file\n"
		result		+= "File:\n"

		for data in self._save_list:
			filename	= data[1]
			result 		+= filename + "\n"

		return result
