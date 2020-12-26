import time
from datetime import datetime
from typing import *
import torch
import torch.optim as optim
from Lib import *


# ----- check environment -----
# env_device = torch.cuda.current_device()
env_device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_device = torch.device(env_device)

print(f"----- Environment -----")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device name:    {torch.cuda.get_device_name(env_device)}")
print(f"Device memory:  {torch.cuda.get_device_properties(env_device).total_memory}")

# ----- model -----
info = ModelInfo()
info.epoch 			= 5
info.batch_size 	= 1
info.learning_rate	= 5e-4
info.train_parameter["Momentum"] = 0.9

info.device_train 	= env_device
info.device_test 	= env_device
info.model 			= FastRCNN_Alexnet()

info.optimizer = optim.SGD(info.model.parameters(), lr=info.learning_rate, momentum=info.train_parameter["Momentum"])
# info.optimizer		= optim.Adam(pack.net.parameters(), lr=1e-4)

# info.scheduler	= optim.lr_scheduler.StepLR(pack.optimizer, step_size=30, gamma=0.1)
info.scheduler = None

print(f"----- Model -----")
print(info.model)

# ----- dataset -----
# data file
file_config_train	= "./Data/train/Data_20201225212039_0.75_1.0_0.1_0.75_320_320.json"
file_config_val		= "./Data/val/Data_20201226131228_0.75_1.0_0.1_0.75_32_32.json"
file_config_test	= "./Data/test/Data_20201223123013_0.5_1.0_0.1_0.5_32_32.json"

data_path_train	= "./Data/train"
data_path_val	= "./Data/val"
data_path_test	= "./Data/test"

# config
config_train 			= Config_Processed()
config_train.file 		= file_config_train
config_train.load()

config_validate 		= Config_Processed()
config_validate.file 	= file_config_val
config_validate.load()

config_test 			= Config_Processed()
config_test.file 		= file_config_test
config_test.load()

# dataset
dataset_train 		= Dataset_Processed(config_train, 		data_path=data_path_train)
dataset_validate 	= Dataset_Processed(config_validate, 	data_path=data_path_val)
dataset_test 		= Dataset_Processed(config_test, 		data_path=data_path_test)

dataset_train.size_positive = 16
dataset_train.size_negative = 16
dataset_validate.size_positive = 16
dataset_validate.size_negative = 16

# dataset list
dataset_list: Dict = {
	"Train": dataset_train,
	"Val": dataset_validate,
	"Test": dataset_test
}

# save to info
info.train_parameter["DataTrain"]	= file_config_train
info.train_parameter["DataVal"]		= file_config_val
info.train_parameter["DataTest"]	= file_config_test

print(f"----- Dataset -----")
print(info.train_parameter["DataTrain"])
print(info.train_parameter["DataVal"])
print(info.train_parameter["DataTest"])

# ----- process -----
now 			= datetime.now()
current_time 	= now.strftime("%Y%m%d%H%M%S")

info.save_path 		= "./Result"
info.save_folder 	= f"Result_{current_time}"

process_hookResult		= TrainProcess_Hook()
process_folderHandler	= TrainProcess_Folder()
process_resultRecord	= TrainProcess_ResultRecord()
process_codeFileSaver	= TrainProcess_PythonFile()
process_dictDataSaver	= TrainProcess_DictSave()
process_resultGraph		= TrainProcess_ResultGraph()

process_folderHandler.stage = [ModelInfo.Stage.TRAIN_END]
process_hookResult.stage 	= [ModelInfo.Stage.TRAIN_END]

process_resultRecord.accuracy_index = 5  # 5 is the confusion matrix of val_total
process_codeFileSaver.add(info.model, 	"FastRCNN_Alexnet.py")
process_dictDataSaver.add(info, 		"ModelInfo.json")

process_resultGraph.addLoss([3, 4, 5], ["Label", "Box", "Total"], "Loss.png")
process_resultGraph.addAccuracy([2, 5], ["Train", "Val"], "Accuracy.png")


# TODO: may move to other place
# Linker Function
def Linker_Hook_execute(stage: int, info_: ModelInfo, data: Dict) -> None:
	process_resultGraph.addConfusionMatrix(
		(process_resultRecord.best_epoch, 3),
		(	["Predict-NIL", "Predict-MOD", "Predict-SEV"],
			["Ground-NIL", "Ground-MOD", "Ground-SEV"]),
		"ConfusionMatrix_Class.png")

	process_resultGraph.addConfusionMatrix(
		(process_resultRecord.best_epoch, 4),
		(	["Predict-MOD-T", "Predict-MOD-F", "Predict-SEV-T", "Predict-SEV-F"],
			["Ground-MOD-T", "Ground-MOD-F", "Ground-SEV-T", "Ground-SEV-F"]),
		"ConfusionMatrix_IOU.png")

	process_resultGraph.addConfusionMatrix(
		(process_resultRecord.best_epoch, 5),
		(	["Predict-NIL", "Predict-MOD", "Predict-SEV", "Predict-IOU-F"],
			["Ground-NIL", "Ground-MOD", "Ground-SEV", "Ground-IOU-F"]),
		"ConfusionMatrix_Total.png")


# when process_resultGraph is called
# it should generate the confusion matrix of best epoch
process_hookResult.func_execute = Linker_Hook_execute

process_codeFileSaver.is_print	= True
process_dictDataSaver.is_print	= True
process_resultRecord.is_print 	= True
process_resultRecord.is_log		= True
process_resultGraph.is_print	= True
process_resultGraph.is_log		= True

info.process_list.append(process_folderHandler)
info.process_list.append(process_resultRecord)
info.process_list.append(process_hookResult)
info.process_list.append(process_resultGraph)
info.process_list.append(process_codeFileSaver)
info.process_list.append(process_dictDataSaver)

# ----- train -----
print(f"----- Train -----")

# config model
info.model.setGradient(FastRCNN_Alexnet.Layer.INPUT_CONV, 	False)
info.model.setGradient(FastRCNN_Alexnet.Layer.ALEXNET, 		False)
info.model.setGradient(FastRCNN_Alexnet.Layer.POOL, 		False)
info.model.setGradient(FastRCNN_Alexnet.Layer.FEATURE, 		True)
info.model.setGradient(FastRCNN_Alexnet.Layer.SOFTMAX, 		True)
info.model.setGradient(FastRCNN_Alexnet.Layer.BOX, 			True)
info.model.is_detach = True

train(dataset_list, info)
