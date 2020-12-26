from typing import *
import os
import torch
from .ModelInfo import TrainProcess, ModelInfo, TrainResultInfo


class TrainProcess_ResultRecord(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.best_epoch:	int		= 0
		self.best_loss:		float	= float("inf")
		self.best_accuracy:	float	= 0.0
		self.best_dict				= None

		self.accuracy_index: int = 0

		self._execute_table: Dict = {
			ModelInfo.Stage.TRAIN_START:			None,
			ModelInfo.Stage.ITERATION_TRAIN_START:	None,
			ModelInfo.Stage.ITERATION_TRAIN_END:	None,
			ModelInfo.Stage.ITERATION_VAL_START:	None,
			ModelInfo.Stage.ITERATION_VAL_END:		self._execute_IterationTestEnd_,
			ModelInfo.Stage.TRAIN_END:				self._execute_TrainEnd_
		}

		self._content_table: Dict = {
			ModelInfo.Stage.TRAIN_START:			None,
			ModelInfo.Stage.ITERATION_TRAIN_START:	None,
			ModelInfo.Stage.ITERATION_TRAIN_END:	None,
			ModelInfo.Stage.ITERATION_VAL_START:	None,
			ModelInfo.Stage.ITERATION_VAL_END:		self._getContent_IterationTestEnd_,
			ModelInfo.Stage.TRAIN_END:				self._getContent_TrainEnd_
		}

		# operation
		# default stage (can be changed by user)
		self.stage.append(ModelInfo.Stage.ITERATION_VAL_END)
		self.stage.append(ModelInfo.Stage.TRAIN_END)

	def __del__(self):
		return

	# Operation
	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		func = self._execute_table[stage]
		if func is None:
			return
		func(info, data)

	def getLogContent(self, stage: int, info: ModelInfo) -> str:
		func = self._content_table[stage]
		if func is None:
			return ""
		return func(info)

	def getPrintContent(self, stage: int, info: ModelInfo) -> str:
		func = self._content_table[stage]
		if func is None:
			return ""
		return func(info)

	# Protected
	def _execute_IterationTestEnd_(self, info: ModelInfo, data: Dict) -> None:
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

	def _execute_TrainEnd_(self, info: ModelInfo, data: Dict) -> None:
		if self.best_dict is None:
			return

		# it is assumed that folder and path must be exist
		path 		= info.save_path
		folder 		= info.save_folder
		folder_path = os.path.join(path, folder)

		# save best dict
		torch.save(self.best_dict, os.path.join(folder_path, "ModelStateDict.tar"))

	def _getContent_IterationTestEnd_(self, info: ModelInfo) -> str:
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

	def _getContent_TrainEnd_(self, info: ModelInfo) -> str:
		content: str = ""
		content 	+= "Operation: save best state dict\n"
		content		+= f"Best: Epoch: {self.best_epoch}; "
		content		+= f"Loss: {self.best_loss:.4f}; "
		content 	+= f"Accuracy: {(self.best_accuracy * 100):.2f}%\n"
		content		+= f"File: ModelStateDict.tar\n"
		return content

