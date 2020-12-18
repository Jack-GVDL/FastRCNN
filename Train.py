import time
from datetime import datetime
from typing import *
import torch
import torch.optim as optim
from Lib import *


# parameter / constant
file_config_train = "./Data/train/Data_20201216100337_0.5_1.0_0.1_0.5.json"
# file_config_train 	= "./Data/train/Data_20201215104746_0.7_0.3.json"
# file_config_train 	= "./Data/train/Data_20201215145930_0.7_0.1.json"
file_config_val = "./Data/val/Data.json"
file_config_test = "./Data/test/Data.json"

data_path_train = "./Data/train"
data_path_val = "./Data/val"
data_path_test = "./Data/test"

# check env
# env_device = torch.cuda.current_device()
env_device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_device = torch.device(env_device)

print(f"----- Environment -----")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device name:    {torch.cuda.get_device_name(env_device)}")
print(f"Device memory:  {torch.cuda.get_device_properties(env_device).total_memory}")

# model
info = ModelInfo()
info.epoch 			= 5
info.batch_size 	= 1
info.learning_rate	= 5e-4
info.train_parameter["Momentum"] 	= 0.9
info.train_parameter["DataTrain"]	= file_config_train
info.train_parameter["DataVal"]		= file_config_val
info.train_parameter["DataTest"]	= file_config_test

info.device_train 	= env_device
info.device_test 	= env_device
info.model 			= FastRCNN()

info.optimizer = optim.SGD(info.model.parameters(), lr=info.learning_rate, momentum=info.train_parameter["Momentum"])
# pack.optimizer		= optim.Adam(pack.net.parameters(), lr=1e-4)
# pack.scheduler	= optim.lr_scheduler.StepLR(pack.optimizer, step_size=30, gamma=0.1)

print(f"----- Model -----")
print(info.model)

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

# dataset list
dataset_list: Dict = {
	"Train": dataset_train,
	"Val": dataset_validate,
	"Test": dataset_test
}

# process
now 			= datetime.now()
current_time 	= now.strftime("%Y%m%d%H%M%S")

info.save_path 		= "./Model"
info.save_folder 	= f"Model_{current_time}"

process_folderHandler	= TrainProcessInfo_FolderHandler()
process_resultRecorder	= TrainProcessInfo_ResultRecorder()
process_codeFileSaver	= TrainProcessInfo_CodeFileSaver()
process_dictDataSaver	= TrainProcessInfo_DictDataSaver()

process_codeFileSaver.add(info.model, 	"Model.py")
process_dictDataSaver.add(info, 		"ModelInfo.json")

process_codeFileSaver.is_print	= True
process_dictDataSaver.is_print	= True
process_resultRecorder.is_print = True
process_resultRecorder.is_log	= True

info.process_list.append(process_folderHandler)
info.process_list.append(process_codeFileSaver)
info.process_list.append(process_dictDataSaver)
info.process_list.append(process_resultRecorder)

# train
print(f"----- Train -----")
time.sleep(0.1)

train(dataset_list, info)
