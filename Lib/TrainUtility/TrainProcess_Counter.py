from typing import *
from .ModelInfo import TrainProcess, ModelInfo


class TrainProcess_Counter(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self._process_list:	List[TrainProcess] = []

		# counter[_][0: cur, 1: target(exclusive)]
		self._counter_list: List[List[int, int]] = []

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	# data
	def setData(self, data: Dict) -> None:
		self._process_list = self._getDataFromDict_(data, "process_list", self._process_list)
		self._counter_list = self._getDataFromDict_(data, "counter_list", self._counter_list)

	def getData(self) -> Dict:
		return {
			"process_list": self._process_list,
			"counter_list": self._counter_list
		}

	# process
	def addProcess(self, process: TrainProcess, counter: int) -> bool:
		self._process_list.append(process)
		self._counter_list.append([0, counter])
		return True

	def rmProcess(self, process: TrainProcess) -> bool:
		try:
			index = self._process_list.index(process)
		except ValueError:
			return False

		self._process_list.pop(index)
		self._counter_list.pop(index)
		return True

	def execute(self, stage: int, info: Any, data: Dict) -> None:
		for index, process in enumerate(self._process_list):

			# add counter
			self._counter_list[index][0] += 1
			if self._counter_list[index][0] < self._counter_list[index][1]:
				continue
			self._counter_list[index][0] = 0

			# run sub-process
			process.execute(stage, info, data)

	def getPrintContent(self, stage: int, info: Any) -> str:
		return "Operation: counter"

	def getLogContent(self, stage: int, info: Any) -> str:
		return "Operation: counter"

	def getInfo(self) -> str:
		return "Counter"
