from typing import *
import os
import shutil
from .Util_Interface import Interface_CodePath
from .TrainProcess import TrainProcess
from .ModelInfo import ModelInfo


class TrainProcess_PythonFile(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.name = "PythonFile"

		self._save_list:	List[Tuple[Interface_CodePath, str]] = []

		# backup
		# # assumed: path is exist
		# # while there is no limitation on whether folder should exist or not
		# self.path:		str	= ""
		# self.folder:	str	= ""

		# operation
		# ...

	def __del__(self):
		return

	# Property
	@property
	def save_list(self):
		return self._save_list.copy()

	# Operation
	# data
	def setData(self, data: Dict) -> None:
		self._save_list = self._getDataFromDict_(data, "save_list", self._save_list)

	def getData(self) -> Dict:
		return {
			"save_list": self._save_list
		}

	# operation
	def addPythonFile(self, obj: Interface_CodePath, filename: str) -> bool:
		self._save_list.append((obj, filename))
		return True

	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		# it is assumed that folder and path must be exist
		path 		= info.save_path
		folder 		= info.save_folder
		folder_path = os.path.join(path, folder)

		# foreach object, save its file
		for data in self._save_list:

			# copy file from src (given by getCodePath()) to dst
			obj			= data[0]
			filename	= data[1]

			# backup
			# TODO: may consider to use ntpath
			# reason why use ntpath
			# https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
			# object_path = obj.getCodePath()
			# filename 	= os.path.basename(object_path)

			shutil.copyfile(obj.getCodePath(), os.path.join(folder_path, filename))

	# info
	def getLogContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	def getPrintContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	def getInfo(self) -> List[List[str]]:
		info: List[List[str]] = []

		# ----- save list -----
		save_list = map(lambda x: ["", x[1]], self._save_list)
		save_list = list(save_list)

		# if the save_list is not empty
		# then the first item will be assigned with a parameter name (save_list)
		if save_list:
			save_list[0][0] = "save_list"

		info.extend(save_list)

		return info

	# Protected
	def _getContent_(self, info: ModelInfo) -> str:
		result: str = ""
		result 		+= "Operation: save code file\n"
		result		+= "File:\n"

		for data in self._save_list:
			obj		= data[0]
			result 	+= obj.getCodePath() + "\n"

		return result
