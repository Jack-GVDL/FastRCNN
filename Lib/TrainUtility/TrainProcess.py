from typing import *
from .Util_Interface import Interface_DictData


class TrainProcess:

	def __init__(self):
		super().__init__()

		# data
		self.name:		str			= "unknown"

		self._stage:	Set[int]	= set()
		self._depend:	Set[Any]	= set()

		self.is_log:	bool		= False
		self.is_print:	bool		= False

		self._process_list:	List[Any] = []

		# operation
		# ...

	def __del__(self):
		return

	# Property
	@property
	def stage(self) -> Set[int]:
		return self._stage.copy()

	@property
	def depend(self) -> Set[int]:
		return self._depend.copy()

	# Operation
	# stage
	def addStage(self, stage: int) -> None:
		self._stage.add(stage)

	def rmStage(self, stage: int) -> None:
		self._stage.remove(stage)

	# dependence
	def addDependence(self, process: Any) -> None:
		self._depend.add(process)

	# TODO: not yet completed
	def rmDependence(self, process: Any) -> None:
		pass

	# data
	def setData(self, data: Dict) -> None:
		pass

	def getData(self) -> Dict:
		return {}

	# backup
	# def setDictData(self, data: Dict) -> None:
	# 	pass
	#
	# def getDictData(self) -> Dict:
	# 	return {}

	# data in variable "data" will be different at different stage
	def execute(self, stage: int, info: Any, data: Dict) -> None:
		raise NotImplementedError

	def getLogContent(self, stage: int, info: Any) -> str:
		return "Process unknown"

	def getPrintContent(self, stage: int, info: Any) -> str:
		return "Process unknown"

	def getInfo(self) -> str:
		return "Process unknown"

	# Protected
	# helper
	# key in any type as there might be string key or int key
	def _getDataFromDict_(self, data: Dict, key: Any, default_value: Any) -> Any:
		if key not in data.keys():
			return default_value
		return data[key]
