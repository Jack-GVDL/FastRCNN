from typing import *
import json
from .Util_Interface import Interface_DictData
from .TrainProcess import TrainProcess
from .ModelInfo import ModelInfo


class TrainProcess_DictLoad(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.name = "DictLoad"

		# file_load should be the full path (either relative or absolute)
		# file type should be json
		self._load_list: List[Tuple[Interface_DictData, str]] = []

		# operation
		# ...

	def __del__(self):
		return

	# Property
	@property
	def load_list(self):
		return self._load_list.copy()

	# Operation
	# data
	def setData(self, data: Dict) -> None:
		self._load_list = self._getDataFromDict_(data, "load_list", self._load_list)

	def getData(self) -> Dict:
		return {
			"load_list": self._load_list
		}

	# operation
	def addDictData(self, obj: Interface_DictData, file_path: str) -> bool:
		self._load_list.append((obj, file_path))
		return True

	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		for data_load in self._load_list:

			# get target object and file path
			# assumed: file and path must exist
			obj			= data_load[0]
			file_path	= data_load[1]

			# load data from json file
			with open(file_path, "r") as f:
				data_json = f.read()

			data_json = json.loads(data_json)

			# save data to target object
			obj.setDictData(data_json)

	# info
	def getInfo(self) -> List[List[str]]:
		info: List[List[str]] = []

		# ----- load list -----
		load_list = map(lambda x: ["", x[1]], self._load_list)
		load_list = list(load_list)

		# only the first one in load_list will be assigned with a parameter name
		if load_list:
			load_list[0][0] = "save_list"

		info.extend(load_list)

		return info
