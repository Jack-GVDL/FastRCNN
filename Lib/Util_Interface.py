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
		raise NotImplementedError
