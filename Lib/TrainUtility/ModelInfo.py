from typing import *
import numpy as np
from .Util_Interface import Interface_DictData
from .TrainProcess import TrainProcessControl


# Data Structure
# basic
class TrainResultInfo:

	def __init__(self, confusion_matrix: np.ndarray, loss: float):
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


class ModelInfo(Interface_DictData):

	# TODO: find a better way
	#  current way will limit the number of stage
	class Stage:
		TRAIN_START:			int = 0
		ITERATION_TRAIN_START:	int = 1
		ITERATION_TRAIN_END:	int = 2
		ITERATION_VAL_START:	int = 3
		ITERATION_VAL_END:		int = 4
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
		self.process_control: TrainProcessControl = TrainProcessControl()

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
		self.process_control.execute(stage, self, data, self.log)

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
			matrix_list: 	List[List[List[int]]] 	= []
			loss_list:		List[float]				= []

			for data in self.result_list[iteration]:
				# confusion matrix
				# assumed: format: confusion_matrix[row][col], where row == col
				matrix:	List[List[int]] = data.confusion_matrix.tolist()
				matrix_list.append(matrix)

				# loss
				loss_list.append(data.loss)

			data_iteration["ConfusionMatrix"]	= matrix_list
			data_iteration["Loss"]				= loss_list

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
		self.epoch 				= self._getDataFromDict_(data, "Epoch", 			self.epoch)
		self.batch_size			= self._getDataFromDict_(data, "BatchSize", 		self.batch_size)
		self.learning_rate		= self._getDataFromDict_(data, "LearningRate", 		self.learning_rate)
		self.model_parameter 	= self._getDataFromDict_(data, "ModelParameter", 	self.model_parameter)
		self.train_parameter 	= self._getDataFromDict_(data, "TrainParameter", 	self.train_parameter)

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
			loss_list	= self._getDataFromDict_(iteration, "Loss", [])

			# assume: len(matrix_list) == len(loss_list)
			for index in range(len(matrix_list)):
				matrix	= np.array(matrix_list[index])
				loss	= loss_list[index]

				info = TrainResultInfo(matrix, loss)
				info_list.append(info)

			# append to result list
			self.result_list.append(info_list)
