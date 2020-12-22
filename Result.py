from typing import *
from Lib import *


# ----- parameter -----
path_model_info: str = "./Model/Model_Test/ModelInfo.json"

# ----- get model info -----
process_dict_data_loader = TrainProcessInfo_DictDataLoader()
info = ModelInfo()

process_dict_data_loader.add(info, path_model_info)
process_dict_data_loader.execute(-1, info, {})

# ----- plot model result -----
# confusion matrix
plotConfusionMatrix(info.result_list[-1][0], ["NIL", "MOD", "SEV"], ["NIL", "MOD", "SEV"])

result_1: List[TrainResultInfo] = map(lambda x: x[0], info.result_list)
result_2: List[TrainResultInfo] = map(lambda x: x[1], info.result_list)

# accuracy and loss
plotAccuracy([result_1, result_2], ["Label", "Box"])
plotLoss([result_1], ["Loss"])
