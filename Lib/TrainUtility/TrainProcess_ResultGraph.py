from typing import *
import os
import matplotlib.pyplot as plt
from .TrainProcess import TrainProcess
from .ModelInfo import ModelInfo, TrainResultInfo
from .Util_Plot import plotLoss, plotAccuracy, plotConfusionMatrix


class TrainProcess_ResultGraph(TrainProcess):

	def __init__(self):
		super().__init__()

		# data
		self.name = "ResultGraph"

		# assume: len(label_list) == len(result[any_index])
		# self.label_list: List[str] = []

		# confusion matrix
		# assume: index in matrix_index_list[_] are all valid
		# assume: len(matrix_index_list) == len(matrix_label_list) == len(matrix_save_list)
		#
		# format
		# matrix_index_list[_][0: iteration, 1: result_index]
		# matrix_label_list[_][0: label_predicted, 1: label_true]
		#
		# where result_index: index of the TrainResultInfo in that iteration
		self.matrix_index_list:		List[Tuple[int, int]] 				= []
		self.matrix_label_list:		List[Tuple[List[str], List[str]]] 	= []
		self.matrix_save_list:		List[str]							= []

		# tendency / line graph
		# select the index of result to print
		# assume: index in loss_index_list and accuracy_index_list are all valid
		self.loss_index_list: 		List[List[int]] = []
		self.loss_label_list:		List[List[str]] = []
		self.loss_save_list:		List[str] 		= []

		self.accuracy_index_list:	List[List[int]] = []
		self.accuracy_label_list:	List[List[str]] = []
		self.accuracy_save_list:	List[str] 		= []

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	# data
	def setData(self, data: Dict) -> None:
		pass

	def getData(self) -> Dict:
		return {}

	# operation
	def addConfusionMatrix(self, index: Tuple[int, int], label: Tuple[List[str], List[str]], save_file: str) -> bool:
		self.matrix_index_list.append(index)
		self.matrix_label_list.append(label)
		self.matrix_save_list.append(save_file)
		return True

	def addLoss(self, index: List[int], label: List[str], save_file: str) -> bool:
		self.loss_index_list.append(index)
		self.loss_label_list.append(label)
		self.loss_save_list.append(save_file)
		return True

	def addAccuracy(self, index: List[int], label: List[str], save_file: str) -> bool:
		self.accuracy_index_list.append(index)
		self.accuracy_label_list.append(label)
		self.accuracy_save_list.append(save_file)
		return True

	def execute(self, stage: int, info: ModelInfo, data: Dict) -> None:
		# reference
		# 1. save figure without part of figure / graph is clipped
		# https://stackoverflow.com/questions/37427362/plt-show-shows-full-graph-but-savefig-is-cropping-the-image/37428142

		# get folder path
		# it is assumed that folder and path must be exist
		path 		= info.save_path
		folder 		= info.save_folder
		folder_path = os.path.join(path, folder)

		# plot
		# - confusion matrix
		# - loss tendency
		# - accuracy tendency
		self._plotConfusionMatrix_(info, folder_path)
		self._plotLoss_(info, folder_path)
		self._plotAccuracy_(info, folder_path)

	# info
	def getLogContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	def getPrintContent(self, stage: int, info: ModelInfo) -> str:
		return self._getContent_(info)

	# TODO: not yet completed
	def getInfo(self) -> List[List[str]]:
		return []

	# Protected
	def _getContent_(self, info: ModelInfo) -> str:
		content: str = ""
		content += f"Operation: save result graph\n"
		content += f"File:\n"

		# add list of filename to content
		for file in self.matrix_save_list:
			content += f"{file}\n"

		for file in self.loss_save_list:
			content += f"{file}\n"

		for file in self.accuracy_save_list:
			content += f"{file}\n"

		return content

	def _plotConfusionMatrix_(self, info: ModelInfo, folder_path: str) -> None:
		for i, index in enumerate(self.matrix_index_list):
			result = info.result_list[index[0]][index[1]]

			plt.clf()
			plotConfusionMatrix(
				result, self.matrix_label_list[i][0], self.matrix_label_list[i][1],
				normalize=True, is_show=False)

			plt.savefig(os.path.join(folder_path, self.matrix_save_list[i]), bbox_inches="tight")
			plt.clf()

	def _plotLoss_(self, info: ModelInfo, folder_path: str) -> None:
		for i, index_list in enumerate(self.loss_index_list):

			# get loss_list
			result_list: List[List[TrainResultInfo]] = []
			for index in index_list:

				temp = list(map(lambda x: x[index], info.result_list))
				result_list.append(temp)

			# plot and save
			plt.clf()
			plotLoss(result_list, self.loss_label_list[i], is_show=False)

			plt.savefig(os.path.join(folder_path, self.loss_save_list[i]), bbox_inches="tight")
			plt.clf()

	def _plotAccuracy_(self, info: ModelInfo, folder_path: str) -> None:
		for i, index_list in enumerate(self.accuracy_index_list):

			# get accuracy_list
			result_list: 	List[List[TrainResultInfo]] = []
			for index in index_list:

				temp = list(map(lambda x: x[index], info.result_list))
				result_list.append(temp)

			# plot and save
			plt.clf()
			plotAccuracy(result_list, self.accuracy_label_list[i], is_show=False)

			plt.savefig(os.path.join(folder_path, self.accuracy_save_list[i]), bbox_inches="tight")
			plt.clf()
