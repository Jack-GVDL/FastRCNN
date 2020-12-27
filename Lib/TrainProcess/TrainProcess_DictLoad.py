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
		self.load_list:	List[Tuple[Interface_DictData, str]] = []

		# operation
		# default stage
		# no default stage, it depends on situation

	def __del__(self):
		return

	# Operation
	def add(self, obj: Interface_DictData, file_path: str) -> bool:
		self.load_list.append((obj, file_path))
		return True

	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		for data_load in self.load_list:

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
