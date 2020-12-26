from typing import *
from Lib import *


# ----- parameter -----
path_model_info: str = "./Result/Result_20201224192614/ModelInfo.json"

# ----- get model info -----
process_dict_data_loader = TrainProcess_DictDataLoader()
info = ModelInfo()

process_dict_data_loader.add(info, path_model_info)
process_dict_data_loader.execute(-1, info, {})

# ----- plot model result -----
# best epoch
best_epoch = 97

# confusion matrix
plotConfusionMatrix(
    info.result_list[best_epoch][3],
    ["Predict-NIL", "Predict-MOD", "Predict-SEV"],
    ["Ground-NIL", "Ground-MOD", "Ground-SEV"],
    normalize=True)

plotConfusionMatrix(
    info.result_list[best_epoch][4],
    ["Predict-MOD-T", "Predict-MOD-F", "Predict-SEV-T", "Predict-SEV-F"],
    ["Ground-MOD-T", "Ground-MOD-F", "Ground-SEV-T", "Ground-SEV-F"],
    normalize=True)

plotConfusionMatrix(
    info.result_list[best_epoch][5],
    ["Predict-NIL", "Predict-MOD", "Predict-SEV", "Predict-IOU-F"],
    ["Ground-NIL", "Ground-MOD", "Ground-SEV", "Ground-IOU-F"],
    normalize=True)

# result
result_train: List[TrainResultInfo] = list(map(lambda x: x[2], info.result_list))
result_label: List[TrainResultInfo] = list(map(lambda x: x[3], info.result_list))
result_box:   List[TrainResultInfo] = list(map(lambda x: x[4], info.result_list))
result_total: List[TrainResultInfo] = list(map(lambda x: x[5], info.result_list))

# accuracy and loss
plotAccuracy([result_label, result_total], ["Label", "Total"])
plotLoss([result_train, result_total], ["Train", "Val"])
