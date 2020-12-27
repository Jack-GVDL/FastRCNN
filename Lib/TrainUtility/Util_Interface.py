from typing import *


class Interface_CodePath:

	def __init__(self):
		super().__init__()

		# data
		# ...

	# Operation
	def getCodePath(self) -> str:
		raise NotImplementedError


class Interface_DictData:

	def __init__(self):
		super().__init__()

		# data
		# ...

	# Operation
	def getDictData(self) -> Dict:
		pass

	def setDictData(self, data: Dict) -> None:
		pass

	# Protected
	# helper
	# key in any type as there might be string key or int key
	def _getDataFromDict_(self, data: Dict, key: Any, default_value: Any) -> Any:
		if key not in data.keys():
			return default_value
		return data[key]
