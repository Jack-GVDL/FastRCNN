from typing import *
from .Util_Interface import Interface_DictData
from .Util_Print import UtilPrint_Table, UtilPrint_Level


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

	@property
	def data(self) -> Dict:
		return self.getData()

	@property
	def process_list(self):
		return self._process_list.copy()

	@data.setter
	def data(self, data: Dict) -> None:
		self.setData(data)

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

	# data in variable "data" will be different at different stage
	def execute(self, stage: int, info: Any, data: Dict) -> None:
		raise NotImplementedError

	# TODO: should log and print function combined
	def getLogContent(self, stage: int, info: Any) -> str:
		return ""

	def getPrintContent(self, stage: int, info: Any) -> str:
		return ""

	def getInfo(self) -> List[List[str]]:
		"""
		it should return the process parameter
		and should not show what is the result of the process execution

		:return: List[Tuple[str, str]], where the
										first item in the tuple is the parameter name and
										second item is the parameter value
		"""
		return []

	# Protected
	# helper
	# key in any type as there might be string key or int key
	def _getDataFromDict_(self, data: Dict, key: Any, default_value: Any) -> Any:
		if key not in data.keys():
			return default_value
		return data[key]


class TrainProcessControl:

	def __init__(self):
		super().__init__()

		# data
		self._process_list: List[TrainProcess] = []

		# format
		# 0: name
		# 1: process_creation_function
		self._template_list: List[Tuple[str, Callable[[], TrainProcess]]] = []

		# operation
		# ...

	def __del__(self):
		return

	# Property
	@property
	def process_list(self) -> List[TrainProcess]:
		return self._process_list.copy()

	@property
	def template_list(self) -> List:
		return self._template_list.copy()

	# Operation
	# backup
	def createProcess(self, name: str) -> TrainProcess:
		# find template based on name
		# it should have only one or zero template
		template: List[Tuple[str, Callable[[], TrainProcess]]] = \
			list(filter(lambda x: (x[0] == name), self._template_list))

		if not template:
			return None

		# create process
		process = template[0][1]()

		# add to process_list
		self.addProcess(process)

		return process

	def destroyProcess(self, process: TrainProcess) -> bool:
		return self.rmProcess(process)

	def addTemplate(self, name: str, func: Callable[[], TrainProcess]) -> bool:
		template = list(filter(lambda x: (x[0] == name), self._template_list))
		if len(template) != 0:
			return False

		self._template_list.append((name, func))
		return True

	# remove template will only remove the template
	# but will not remove any process in the process_list
	def rmTemplate(self, name: str) -> bool:
		# find the target template tuple by name
		index: int = -1
		for i, item in enumerate(self._template_list):
			if item[0] != name:
				continue
			index = i
			break

		# check if target exist or not
		# if absent, then do nothing
		if index == -1:
			return False

		self._template_list.pop(index)
		return True

	def addProcess(self, process: TrainProcess) -> bool:
		self._process_list.append(process)
		return True

	def rmProcess(self, process: TrainProcess) -> bool:
		try:
			index: int = self._process_list.index(process)
		except ValueError:
			return False

		self._process_list.pop(index)
		return True

	def execute(self, stage: int, info: Any, data: Dict, log: List[str]) -> None:
		# get process that needed to be executed
		process_list = filter(lambda x: (stage in x.stage), self._process_list)

		# foreach process
		for process in process_list:
			process.execute(stage, info, data)

			# logging
			if process.is_log:
				log.append(process.getLogContent(stage, info))

			# print to screen (to stdout)
			if process.is_print:
				print(process.getPrintContent(stage, info))


class TrainProcessProbe(TrainProcess):

	class ProcessNode:

		def __init__(self):
			# data
			self.process:		TrainProcess	= None
			self.process_list:	Dict[int, Any] 	= None

	def __init__(self):
		super().__init__()

		# data
		self.process_control: TrainProcessControl = None

		# template info
		# list of template name
		self.info_template: List[str] = []

		# process info
		self.info_process: Dict[int, List[TrainProcess]] = {}

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	# data
	def setData(self, data: Dict) -> None:
		self.process_control = self._getDataFromDict_(data, "process_control", self.process_control)

	def getData(self) -> Dict:
		return {
			"process_control":	self.process_control,
			"info_template": 	self.info_template,
			"info_process":		self.info_process
		}

	# operation
	def execute(self, stage: int, info: Any, data: Dict) -> None:
		self.probe()

	def probe(self) -> bool:
		if self.process_control is None:
			return False

		# probe
		self._probeTemplate_()
		self._probeProcess_()

		return True

	# info
	def getPrintContent(self, stage: int, info: Any) -> str:
		return self._getContent_(info)

	def getLogContent(self, stage: int, info: Any) -> str:
		return self._getContent_(info)

	def getInfo(self) -> List[List[str]]:
		return [
			["process_control", "exist" if self.process_control is not None else "None"]
		]

	# Protected
	# probe
	def _probeTemplate_(self) -> None:
		template_list: List = self.process_control.template_list
		self.info_template	= list(map(lambda x: x[0], template_list))

	def _probeProcess_(self) -> None:
		process_list = self.process_control.process_list

		for process in process_list:
			for stage in process.stage:

				# check if stage exist in info_process
				# if not, create the key and put a list in it
				if stage not in self.info_process.keys():
					self.info_process[stage] = []

				self.info_process[stage].append(process)

	# content
	def _getContent_(self, info: Any) -> str:
		content: str = ""
		content += "Operation: probe\n"
		content += self._getContent_Template_()
		content += self._getContent_Process_()
		return content

	def _getContent_Template_(self) -> str:
		content: str = ""

		for template in self.info_template:
			content += template
			content += "\n"

		content += "\n"
		return content

	def _getContent_Process_(self) -> str:
		content: str = ""

		# ----- util print -----
		# level
		util_level = UtilPrint_Level()
		util_level.indent = 4

		# table
		util_table = UtilPrint_Table()
		util_table.padding_inner = 1
		util_table.padding_outer = 0
		util_table.addColumn("Parameter")
		util_table.addColumn("Value")

		# ----- get content -----
		for stage in self.info_process.keys():
			content += f"Stage: {stage}\n"

			# indent
			util_level.incrementLevel()

			for process in self.info_process[stage]:
				content += util_level.createIndent()
				content += process.name
				content += '\n'

				# check if getInfo return non-empty list
				# if is empty list, then print nothing
				# else, print the info table
				info: List[List[str, str]] = process.getInfo()
				if not info:
					# newline for next process
					content += '\n'
					continue

				util_table.resetRow()
				util_table.extendRow(info)
				content += util_table.createTable(util_level)

				# newline for next process
				content += '\n'

			# indent
			util_level.decrementLevel()

		content += "\n"
		return content
