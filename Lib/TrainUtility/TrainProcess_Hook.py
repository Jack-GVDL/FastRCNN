from typing import *
from .ModelInfo import TrainProcess, ModelInfo, TrainResultInfo


class TrainProcess_Hook(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.func_execute:			Callable[[int, ModelInfo, Dict], None] 	= None
		self.func_log_content:		Callable[[int, ModelInfo], str]			= None
		self.func_print_content:	Callable[[int, ModelInfo], str]			= None

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	def setData(self, data: Dict) -> None:
		self.func_execute 		= self._getDataFromDict_(data, "func_execute", 			self.func_execute)
		self.func_log_content 	= self._getDataFromDict_(data, "func_log_content", 		self.func_log_content)
		self.func_print_content = self._getDataFromDict_(data, "func_print_content", 	self.func_print_content)

	def getData(self) -> Dict:
		return {
			"func_execute": 		self.func_execute,
			"func_log_content":		self.func_log_content,
			"func_print_content":	self.func_print_content
		}

	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		if self.func_execute is None:
			return
		self.func_execute(stage, info, data)

	def getLogContent(self, stage: int, info: ModelInfo) -> str:
		if self.func_log_content is None:
			return ""
		return self.func_log_content(stage, info)

	def getPrintContent(self, stage: int, info: ModelInfo) -> str:
		if self.func_print_content is None:
			return ""
		return self.func_print_content(stage, info)