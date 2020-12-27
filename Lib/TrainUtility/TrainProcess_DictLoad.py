from typing import *
import json
from .Util_Interface import Interface_DictData
from .ModelInfo import TrainProcess, ModelInfo


class TrainProcess_DictLoad(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		# file_load should be the full path (either relative or absolute)
		# file type should be json
		self._load_list:	List[Tuple[Interface_DictData, str]] = []

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	def setData(self, data: Dict) -> None:
		self._load_list = self._getDataFromDict_(data, "load_list", self._load_list)

	def getData(self) -> Dict:
		return {
			"load_list": self._load_list
		}

	def add(self, obj: Interface_DictData, file_path: str) -> bool:
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
