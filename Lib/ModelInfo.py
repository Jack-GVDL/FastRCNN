import json
import shutil
from datetime import datetime
from typing import *
import os
import ntpath
import numpy as np
import torch
from .Util_Interface import Interface_CodePath, Interface_DictData


# Data Structure
# basic
class TrainResultInfo:

	def __init__(self, confusion_matrix: np.ndarray, loss: float = 0.0):
		super().__init__()

		# data
		# accuracy, precision, recall and f1_score can be obtained from confusion matrix
		# all the variable below should have the value [0.0, 0.1]
		# self.accuracy:	float = 0.0
		# self.precision:	float = 0.0
		# self.recall:	float = 0.0
		# self.f1_score:	float = 0.0

		# confusion matrix
		# row: actual / ground truth
		# col: prediction
		# confusion_matrix[row][col]
		self.confusion_matrix: np.ndarray = confusion_matrix

		self.loss: float = loss

		# operation
		# ...

	def __del__(self):
		return

	# Property
	# ...

	# Operation
	# (TP + TN) / (TP + FP + FN + TN)
	def getAccuracy(self) -> float:
		assert self.confusion_matrix is not None

		# check if there is sample in the matrix
		# if no sample, just output 0
		size_sample = np.sum(self.confusion_matrix)
		if size_sample == 0:
			return 0.0

		# count TP and TN, i.e. item in the diagonal
		matrix_diagonal = np.diagonal(self.confusion_matrix)

		return np.sum(matrix_diagonal) / size_sample

	# TP / (TP + FP)
	def getPrecision(self, positive_list: np.ndarray) -> float:
		assert self.confusion_matrix is not None

		# first sort list
		# or may get wrong item in matrix[index][label]
		positive_list = np.sort(positive_list)

		# select row
		# then check if the resultant matrix is empty or not
		matrix = self.confusion_matrix[positive_list, :]
		if np.sum(matrix) == 0:
			return 0.0

		count_true = 0
		for index, label in enumerate(positive_list):
			count_true += matrix[index][label]

		return count_true / np.sum(matrix)

	# TP / (TP + FN)
	def getRecall(self, positive_list: np.ndarray) -> float:
		assert self.confusion_matrix is not None

		# first sort list
		# or may get wrong item in matrix[label][index]
		positive_list = np.sort(positive_list)

		# select col
		# then check if the resultant matrix is empty or not
		matrix = self.confusion_matrix[:, positive_list]
		if np.sum(matrix) == 0:
			return 0.0

		count_true = 0
		for index, label in enumerate(positive_list):
			count_true += matrix[label][index]

		return count_true / np.sum(matrix)

	# 2 * (Recall * Precision) / (Recall + Precision)
	def getF1Score(self, positive_list: np.ndarray) -> float:
		assert self.confusion_matrix is not None

		recall		= self.getRecall(positive_list)
		precision	= self.getPrecision(positive_list)

		if recall + precision == 0:
			return 0.0
		return 2 * (recall * precision) / (recall + precision)


class TrainProcessInfo:

	def __init__(self):
		super().__init__()

		# data
		self.stage:		List[int]	= []
		self.is_log:	bool		= False
		self.is_print:	bool		= False

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	# data in variable "data" will be different at different stage
	def execute(self, stage: int, info: Any, data: Dict) -> None:
		raise NotImplementedError

	def getLogContent(self, stage: int, info: Any) -> str:
		return "Process unknown"

	def getPrintContent(self, stage: int, info: Any) -> str:
		return "Process unknown"


class ModelInfo(Interface_DictData):

	# TODO: find a better way
	#  current way will limit the number of stage
	class Stage:
		TRAIN_START:			int = 0
		ITERATION_TRAIN_START:	int = 1
		ITERATION_TRAIN_END:	int = 2
		ITERATION_TEST_START:	int = 3
		ITERATION_TEST_END:		int = 4
		TRAIN_END:				int = 5

	def __init__(self):
		super().__init__()

		# data
		# object
		self.model 		= None
		self.optimizer 	= None
		self.scheduler	= None

		self.device_train	= None
		self.device_test	= None

		# model parameter
		self.model_parameter:	Dict	= {}

		# train parameter
		self.epoch:				int		= 100
		self.batch_size: 		int		= 2
		self.learning_rate:		float	= 0.0
		self.train_parameter: 	Dict 	= {}  # other train parameter (e.g. momentum in SGD)

		# TODO: not yet put in use
		# result recorder
		# result will only be generated if result_generator is present
		# result_generator should be a function
		# TODO: result_generator can be the type of TrainProcessInfo
		self.result_generator: Callable[[Any], None] = None

		# result_list[iteration][result_index]
		# result_index indicates that there may be multiple result in the same iteration
		self.result_list: List[List[TrainResultInfo]] = []

		# pre/post processing
		self.process_list: List[TrainProcessInfo] = []

		# data will be changed in the operation
		self.iteration:	int = 0

		# text log
		self.log: List[str] = []

		# save
		# all the stuff that needed to be saved should be save here
		self.save_path:		str = ""
		self.save_folder:	str = ""

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	def executeProcess(self, stage: int, data: Dict) -> None:
		# get process that needed to be executed
		process_list = filter(lambda x: (stage in x.stage), self.process_list)

		# foreach process
		for process in process_list:
			process.execute(stage, self, data)

			# logging
			if process.is_log:
				self.log.append(process.getLogContent(stage, self))

			# print to screen (to stdout)
			if process.is_print:
				print(process.getPrintContent(stage, self))

	def getDictData(self) -> Dict:
		iteration_list: List[Dict] = []

		# generate iteration data
		# current, data for each iteration contain
		# - confusion matrix
		# - loss
		for iteration in range(len(self.result_list)):
			data_iteration: Dict = {}

			# ----- confusion matrix -----
			# get confusion matrix
			# matrix_list[data_index][row][col]
			matrix_list: List[List[List[int]]] = []

			for data in self.result_list[iteration]:
				# assumed: format: confusion_matrix[row][col], where row == col
				matrix:	List[List[int]] = data.confusion_matrix.tolist()
				matrix_list.append(matrix)

			data_iteration["ConfusionMatrix"] = matrix_list

			# ----- loss -----
			# only store the loss in the first (the most important) result info
			data_iteration["Loss"] = self.result_list[iteration][0].loss

			# append to iteration list
			iteration_list.append(data_iteration)

		return {
			# train parameter
			"Epoch":			self.epoch,
			"BatchSize":		self.batch_size,
			"LearningRate":		self.learning_rate,
			"ModelParameter":	self.model_parameter,
			"TrainParameter":	self.train_parameter,

			# data of each iteration / epoch
			"IterationData":	iteration_list,

			# log
			"Log":				self.log
		}

	def setDictData(self, data: Dict) -> None:
		# train parameter
		self.epoch 				= self._getDataFromDict_(data, "Epoch", self.epoch)
		self.batch_size			= self._getDataFromDict_(data, "BatchSize", self.batch_size)
		self.learning_rate		= self._getDataFromDict_(data, "LearningRate", self.learning_rate)
		self.model_parameter 	= self._getDataFromDict_(data, "ModelParameter", self.model_parameter)
		self.train_parameter 	= self._getDataFromDict_(data, "TrainParameter", self.train_parameter)

		# log
		# do we need it ?
		self.log.clear()
		self.log = self._getDataFromDict_(data, "Log", self.log)

		# data of each iteration / epoch
		# do we need it ?
		# clear the previous data in result_list
		self.result_list.clear()

		# get iteration list from dict
		iteration_list = self._getDataFromDict_(data, "IterationData", [])

		for iteration in iteration_list:
			info_list: List[TrainResultInfo] = []

			# get confusion matrix
			matrix_list = self._getDataFromDict_(iteration, "ConfusionMatrix", np.zeros((0, 0)))

			for data_matrix in matrix_list:
				matrix	= np.array(data_matrix)
				info	= TrainResultInfo(matrix)
				info_list.append(info)

			# set loss
			info_list[0].loss = self._getDataFromDict_(iteration, "Loss", 0.0)

			# append to result list
			self.result_list.append(info_list)


# train process info
class TrainProcessInfo_Hook(TrainProcessInfo):

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


class TrainProcessInfo_FolderHandler(TrainProcessInfo):

	def __init__(self):
		super().__init__()

		# data
		# ...

		# operation
		# default stage (can be changed by user)
		self.stage.append(ModelInfo.Stage.TRAIN_START)

	def __del__(self):
		return

	# Operation
	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		path 	= info.save_path
		folder 	= info.save_folder

		# check if folder exist or not
		# if not exist, then create one
		folder_path = os.path.join(path, folder)
		if not os.path.isdir(folder_path):
			os.mkdir(folder_path)


class TrainProcessInfo_CodeFileSaver(TrainProcessInfo):

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


class TrainProcessInfo_DictDataLoader(TrainProcessInfo):

	def __init__(self):
		super().__init__()

		# data
		# file_load should be the full path (either relative or absolute)
		# file type should be json
		self.load_list:	List[Tuple[Interface_DictData, str]] = []

		# operation
		# default stage
		# no default stage, it depends on situation

	def __del__(self):
		return

	# Operation
	def add(self, obj: Interface_DictData, file_path: str) -> bool:
		self.load_list.append((obj, file_path))
		return True

	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		for data_load in self.load_list:

			# get target object and file path
			# assumed: file and path must exist
			obj			= data_load[0]
			file_path	= data_load[1]

			# load data from json file
			with open(file_path, "r") as f:
				data_json = f.read()

			data_json = json.loads(data_json)

			# save data to target object
			obj.setDictData(data_json)


class TrainProcessInfo_DictDataSaver(TrainProcessInfo):

	def __init__(self):
		super().__init__()

		# data
		self.save_list:	List[Tuple[Interface_DictData, str]] = []

		# operation
		# default stage (can be changed by user)
		self.stage.append(ModelInfo.Stage.TRAIN_END)

	def __del__(self):
		return

	# Operation
	def add(self, obj: Interface_DictData, filename: str) -> bool:
		self.save_list.append((obj, filename))
		return True

	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		# it is assumed that folder and path must be exist
		path 		= info.save_path
		folder 		= info.save_folder
		folder_path = os.path.join(path, folder)

		# foreach object, save its file
		for data_save in self.save_list:

			# convert data from dict to json file
			obj			= data_save[0]
			filename	= data_save[1]

			data_dict	= obj.getDictData()
			data_json	= json.dumps(data_dict, indent=4)

			# save to dst folder with given filename
			with open(os.path.join(folder_path, filename), "w") as f:
				f.write(data_json)

	def getLogContent(self, stage: int, info: Any) -> str:
		return self._getContent_(info)

	def getPrintContent(self, stage: int, info: Any) -> str:
		return self._getContent_(info)

	# Protected
	def _getContent_(self, info: ModelInfo) -> str:
		result: str = ""
		result 		+= "Operation: save code file\n"
		result		+= "File:\n"

		for data in self.save_list:
			filename	= data[1]
			result 		+= filename + "\n"

		return result


class TrainProcessInfo_ResultRecorder(TrainProcessInfo):

	def __init__(self):
		super().__init__()

		# data
		self.best_epoch:	int		= 0
		self.best_loss:		float	= float("inf")
		self.best_accuracy:	float	= 0.0
		self.best_dict				= None

		self._execute_table: Dict = {
			ModelInfo.Stage.TRAIN_START:			None,
			ModelInfo.Stage.ITERATION_TRAIN_START:	None,
			ModelInfo.Stage.ITERATION_TRAIN_END:	None,
			ModelInfo.Stage.ITERATION_TEST_START:	None,
			ModelInfo.Stage.ITERATION_TEST_END:		self._execute_IterationTestEnd_,
			ModelInfo.Stage.TRAIN_END:				self._execute_TrainEnd_
		}

		self._content_table: Dict = {
			ModelInfo.Stage.TRAIN_START:			None,
			ModelInfo.Stage.ITERATION_TRAIN_START:	None,
			ModelInfo.Stage.ITERATION_TRAIN_END:	None,
			ModelInfo.Stage.ITERATION_TEST_START:	None,
			ModelInfo.Stage.ITERATION_TEST_END:		self._getContent_IterationTestEnd_,
			ModelInfo.Stage.TRAIN_END:				self._getContent_TrainEnd_
		}

		# operation
		# default stage (can be changed by user)
		self.stage.append(ModelInfo.Stage.ITERATION_TEST_END)
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
		# currently use the first result data in the epoch (which should be the major / most important result)
		result: TrainResultInfo = info.result_list[-1][0]

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
		content += f"Loss: {result_list[0].loss:.5f}; "

		# there may be multiple accuracy
		# just print it one-by-one
		content += f"Accuracy: "
		for i, result in enumerate(result_list):

			content += f"{(result.getAccuracy() * 100):.2f}%"
			if i != len(result_list) - 1:
				content += ", "

		return content

	def _getContent_TrainEnd_(self, info: ModelInfo) -> str:
		result: str = ""
		result 		+= "Operation: save best state dict\n"
		result		+= f"Best: Epoch: {self.best_epoch}; "
		result		+= f"Loss: {self.best_loss:.4f}; "
		result 		+= f"Accuracy: {(self.best_accuracy* 100):.2f}%\n"
		result		+= f"File: ModelStateDict.tar\n"
		return result
