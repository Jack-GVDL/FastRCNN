from typing import *
import os
import torch
from .TrainProcess import TrainProcess
from .ModelInfo import ModelInfo, TrainResultInfo


class TrainProcess_ResultData(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.name = "ResultData"

		self.best_epoch:	int		= 0
		self.best_loss:		float	= float("inf")
		self.best_accuracy:	float	= 0.0
		self.best_dict				= None

		self.accuracy_index: int = 0

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	# data
	def setData(self, data: Dict) -> None:
		self.accuracy_index = self._getDataFromDict_(data, "accuracy_index", self.accuracy_index)

	def getData(self) -> Dict:
		return {
			"best_epoch": 		self.best_epoch,
			"best_loss":		self.best_loss,
			"best_accuracy":	self.best_accuracy,
			"best_dict":		self.best_dict,
			"accuracy_index":	self.accuracy_index
		}

	# operation
	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		# get the most recent result
		if not info.result_list:
			return

		# it is assumed that if certain iteration / epoch occur
		# then there must be at least one result data
		# choose the compare accuracy using self.accuracy_index
		result: TrainResultInfo = info.result_list[-1][self.accuracy_index]

		# record the dict of best result
		# current result parameter is the accuracy
		accuracy = result.getAccuracy()
		if accuracy < self.best_accuracy:
			return

		self.best_epoch 	= info.iteration
		self.best_loss		= result.loss
		self.best_accuracy	= accuracy
		self.best_dict		= info.model.state_dict()

	# info
	def getLogContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	def getPrintContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	def getInfo(self) -> List[List[str]]:
		return [
			["accuracy index", str(self.accuracy_index)]
		]

	# Protected
	def _getContent_(self, info: ModelInfo) -> str:
		# get the most recent result
		if not info.result_list:
			return ""
		result_list: List[TrainResultInfo] = info.result_list[-1]

		content: str = ""
		content += f"Epoch: {info.iteration}; "

		# it should have only one loss (or should it?), so using the first one is ok
		content += f"Loss: {result_list[self.accuracy_index].loss:.5f}; "

		# there may be multiple accuracy
		# just print it one-by-one
		content += f"Accuracy: "
		for i, result in enumerate(result_list):

			content += f"{(result.getAccuracy() * 100):.2f}%"
			if i != len(result_list) - 1:
				content += ", "

		return content


class TrainProcess_ResultRecord(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.name = "ResultRecord"

		self.result: TrainProcess_ResultData = None

		# file name of the saved target
		self.file_save:	str = "ModelStateDict.tar"

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	# data
	def setData(self, data: Dict) -> None:
		self.result 	= self._getDataFromDict_(data, "result", self.result)
		self.file_save 	= self._getDataFromDict_(data, "file_save", self.file_save)

	def getData(self) -> Dict:
		return {
			"result": 		self.result,
			"file_save":	self.file_save
		}

	# operation
	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		if self.result is None:
			return
		if self.result.best_dict is None:
			return

		# it is assumed that folder and path must be exist
		path 		= info.save_path
		folder 		= info.save_folder
		folder_path = os.path.join(path, folder)

		# save best dict
		torch.save(self.result.best_dict, os.path.join(folder_path, self.file_save))

	# info
	def getLogContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	def getPrintContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	def getInfo(self) -> List[List[str]]:
		return [
			["file_save", self.file_save]
		]

	# Protected
	def _getContent_(self, info: ModelInfo) -> str:
		content: str = ""
		content += "Operation: save best state dict\n"

		# check if target is present or not
		if self.result is None:
			content += "Result is absent"
			return content
		if self.result.best_dict is None:
			content += "Best dict is absent"
			return content

		content += f"Best: Epoch: {self.result.best_epoch}; "
		content += f"Loss: {self.result.best_loss:.4f}; "
		content += f"Accuracy: {(self.result.best_accuracy * 100):.2f}%\n"
		content += f"File: ModelStateDict.tar\n"

		return content
