import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from Lib import *
from Lib.Util import getCenterBox, convert_xywh_x1y1x2y2, getTopLeftBox, scaleBox_xywh, \
	convert_x1y1x2y2_xywh, clipBox


# ---- check environment -----
# env_device = torch.cuda.current_device()
env_device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_device = torch.device(env_device)

print(f"----- Environment -----")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device name:    {torch.cuda.get_device_name(env_device)}")
print(f"Device memory:  {torch.cuda.get_device_properties(env_device).total_memory}")


# ----- model -----
# it is testing so ModelInfo is not used
model = FastRCNN_Alexnet()

# load model
state_dict = "./Result/Result_20201226131730/ModelStateDict.tar"
model.load_state_dict(torch.load(state_dict))

# config
model.train(False)
# model.to(env_device)
model.cpu()

print(f"----- Model -----")
print(model)


# ----- dataset -----
file_config_test = "./Data/train/DataRaw.json"
data_path_test = "./Data/train"

config_test = Config_Processed()
config_test.file = file_config_test
config_test.load()

# dataset
dataset_test = Dataset_Processed(config_test, data_path=data_path_test)


# foreach image
for i in range(len(dataset_test)):
	# get image
	data = dataset_test[i]
	data_image = data[Dataset_Processed.Label.IMAGE_LIST][0].numpy()

	# selective search
	# then output roi with format xywh
	size_grid = (20, 20)

	roi = runSelectiveSearch(data_image, 10, size_grid)
	roi = convert_x1y1x2y2_xywh(roi)
	roi = clipBox(roi, (839, 839))

	# roi = []
	# for y in range(10):
	# 	for x in range(10):
	# 		roi.append([x * 84, y * 84, 84, 84])
	# roi = np.array(roi)

	# get region captured
	(box_mod, box_sev) = runRegionCapture(model, data_image, roi.copy())

	# box ground truth
	box_ground = data[Dataset_Processed.Label.BOX_LIST]
	box_ground = scaleBox_xywh(box_ground, (840, 840))

	# plot graph
	plotImageBox(data_image[0:1, :, :], [box_ground, box_mod, box_sev], ["Ground", "MOD", "SEV"], ["green", "orange", "red"])

	# breakpoint
	breakpoint()
