from typing import *
from .TrainProcess import TrainProcess
from .ModelInfo import ModelInfo


class TrainProcess_Hook(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.name = "Hook"

		self.func_execute:			Callable[[int, ModelInfo, Dict], None] 	= None
		self.func_log_content:		Callable[[int, ModelInfo], str]			= None
		self.func_print_content:	Callable[[int, ModelInfo], str]			= None

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	# data
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

	# operation
	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		if self.func_execute is None:
			return
		self.func_execute(stage, info, data)

	# info
	def getLogContent(self, stage: int, info: ModelInfo) -> str:
		if self.func_log_content is None:
			return ""
		return self.func_log_content(stage, info)

	def getPrintContent(self, stage: int, info: ModelInfo) -> str:
		if self.func_print_content is None:
			return ""
		return self.func_print_content(stage, info)

	def getInfo(self) -> List[List[str]]:
		return [
			["func_execute", 		"exist" if self.func_execute is not None else "None"],
			["func_log_content", 	"exist" if self.func_log_content is not None else "None"],
			["func_print_content", 	"exist" if self.func_print_content is not None else "None"]
		]
