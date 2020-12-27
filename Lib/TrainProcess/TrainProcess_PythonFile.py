from typing import *
import os
import shutil
from .Util_Interface import Interface_CodePath
from .ModelInfo import TrainProcess, ModelInfo


class TrainProcess_PythonFile(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.save_list:	List[Tuple[Interface_CodePath, str]] = []

		# assumed: path is exist
		# while there is no limitation on whether folder should exist or not
		self.path:		str	= ""
		self.folder:	str	= ""

		# operation
		# default stage (can be changed by user)
		self.stage.append(ModelInfo.Stage.TRAIN_END)

	def __del__(self):
		return

	# Operation
	def add(self, obj: Interface_CodePath, filename: str) -> bool:
		self.save_list.append((obj, filename))
		return True

	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		# it is assumed that folder and path must be exist
		path 		= info.save_path
		folder 		= info.save_folder
		folder_path = os.path.join(path, folder)

		# foreach object, save its file
		for data in self.save_list:

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

	def getLogContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	def getPrintContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	# Protected
	def _getContent_(self, info: ModelInfo) -> str:
		result: str = ""
		result 		+= "Operation: save code file\n"
		result		+= "File:\n"

		for data in self.save_list:
			obj		= data[0]
			result 	+= obj.getCodePath() + "\n"

		return result
