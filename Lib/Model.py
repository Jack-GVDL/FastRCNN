from typing import *
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from Lib.Util import convert_xywh_x1y1x2y2
from Lib.Util_Interface import Interface_CodePath


# Data Structure
class FastRCNN(
	nn.Module,
	Interface_CodePath):

	class Layer:
		INPUT_CONV:	int = 0
		ALEXNET:	int = 1
		POOL:		int = 2
		FEATURE:	int = 3
		SOFTMAX:	int = 4
		BOX:		int = 5

	def __init__(self):
		super().__init__()

		# hyper-parameter / constant
		pool_size = (13, 13)

		# alexnet input channel number is 3
		# but OUR image channel number is 4
		# so an additional layer is required
		#
		# it is assume that all channel is equally important
		# if not, then other method is required to use
		self.input_conv = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=(1, 1))

		# ----- pretrained AlexNet -----
		# remove the last layer in the features
		alexnet = models.alexnet(pretrained=True)
		self.alexnet_seq = nn.Sequential(*list(alexnet.features.children())[:-1])

		# ----- ROI Pooling Layer -----
		# max-pooling 130*130 ROI to 13*13 output grid cell
		self.roi_pool5 = torchvision.ops.RoIPool(output_size=pool_size, spatial_scale=0.059594755)

		# ----- ROI Feature Vector -----
		# the input feature is 43264 as 43264 = 256 (layer) * 13 (H) * 13 (W)
		size_proposal	= 256 * pool_size[0] * pool_size[1]
		# size_feature	= 256 * pool_size[0] * pool_size[1]

		# self-define
		self.feature = nn.Sequential(
			nn.Linear(	in_features=size_proposal, out_features=4096),
			# nn.Linear(	in_features=size_feature + size_proposal, out_features=4096),
			nn.ReLU(	inplace=False),
			nn.Dropout(	inplace=False),

			nn.Linear(	in_features=4096, out_features=4096),
			nn.ReLU(	inplace=False),
			# nn.Dropout(	inplace=False)
		)

		# alexnet.classifier
		# self.feature = nn.Sequential(
		# 	nn.Linear(in_features=size_proposal, out_features=4096),
		# 	*list(alexnet.classifier.children())[2:-1]
		# )

		# ----- Softmax and BBox Regressor -----
		cls_score_fc = nn.Linear(in_features=4096, out_features=3, bias=True)
		torch.nn.init.normal_(cls_score_fc.weight, std=0.01)

		self.cls_score = nn.Sequential(
			cls_score_fc,

			# reason why not use softmax here
			# https://discuss.pytorch.org/t/cross-entropy-loss-is-not-decreasing/43814/3
			# nn.Softmax(dim=1)
		)

		# 3 bbox (each of size is 4) is required
		box_pred_fc = nn.Linear(in_features=4096, out_features=3 * 4, bias=True)
		torch.nn.init.normal_(box_pred_fc.weight, std=0.001)

		self.box_pred = nn.Sequential(
			box_pred_fc,
		)

		# ----- Loss -----
		# simple method
		self.cross_entropy_loss = nn.CrossEntropyLoss()
		self.smooth_l1_loss		= nn.SmoothL1Loss()

		# complex method
		# self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
		# self.smooth_l1_loss		= nn.SmoothL1Loss(reduction="none")

		# ----- Layer -----
		self.layer:	Dict = {
			self.Layer.INPUT_CONV:	self.input_conv,
			self.Layer.ALEXNET:		self.alexnet_seq,
			self.Layer.POOL:		self.roi_pool5,
			self.Layer.FEATURE:		self.feature,
			self.Layer.SOFTMAX:		self.cls_score,
			self.Layer.BOX:			self.box_pred
		}

	def forward(self, image_list, roi_list, roi_index_list):
		x = image_list

		# input
		x = self.input_conv(x)

		# alexnet
		x = self.alexnet_seq(x)

		# roi pooling
		# convert roi from xywh to x1y1x2y2
		# roi_list = roi_list.clone().detach()

		roi_list 		= convert_xywh_x1y1x2y2(roi_list)
		roi_list 		= torch.cat((roi_index_list.view(-1, 1), roi_list), 1)

		# feature_list 	= torch.tensor([[0, 0, 839, 839] for _ in range(roi_index_list.shape[0])], dtype=torch.float).cuda()
		# feature_list	= torch.cat((roi_index_list.view(-1, 1), feature_list), 1)

		proposal 	= self.roi_pool5(x, roi_list)
		# feature 	= self.roi_pool5(x, feature_list)

		# merge proposal and feature map into vector
		# x = torch.cat((
		# 	proposal.view(proposal.shape[0], -1),
		# 	feature.view(feature.shape[0], -1)),
		# 	1)

		# only use proposal map
		x = proposal

		# backward propagation should be stopped here
		x = x.detach()

		# roi feature
		x = x.view(x.shape[0], -1)
		x = self.feature(x)

		# softmax and bbox regressor
		cls_score 	= self.cls_score(x)
		box			= self.box_pred(x).view((-1, 3, 4))

		return cls_score, box

	def criterion(self, predict_class, predict_box, y_class, y_box):
		print(predict_class.shape)
		print(predict_box.shape)
		print(y_class.shape)
		print(y_box.shape)
		breakpoint()

		# ----- label -----
		loss_class = self.cross_entropy_loss(predict_class, y_class)

		# ----- box -----
		# masking
		# bbox is ignored if u == 0 (label: background)
		# [0, 0, 0, 0] for bbox from u == 0
		# [1, 1, 1, 1] for bbox from u >= 1
		mask	= (y_class != 0).float().view(-1, 1).expand((-1, 4))
		label	= y_class.view(-1, 1, 1).expand(y_class.size(0), 1, 4)

		# smooth l1 loss
		# smooth l1 loss is used as described in
		# https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html
		loss_box = self.smooth_l1_loss(predict_box.gather(1, label).squeeze(1) * mask, y_box * mask)

		# this is the sample method
		imbalance_class = 1.0
		imbalance_box   = 1.0

		loss_total = imbalance_class * loss_class + imbalance_box * loss_box
		return loss_total, loss_class, loss_box

		# this is the complex method and not yet verify
		# loss_box = torch.sum(loss_box, 1)
		# # (loss_box, _) = torch.min(loss_box, 1)
		#
		# # multi-task loss
		# imbalance	= 1
		# loss_total	= loss_class + imbalance * loss_box
		#
		# return loss_total.mean(), loss_class.mean(), loss_box.mean()

	def setGradient(self, layer: Layer, is_require_gradient: bool) -> None:
		# get layer
		target_layer = self.layer[layer]

		# https://stackoverflow.com/questions/51638932/how-do-you-change-require-grad-to-false-for-each-parameters-in-your-model-in-pyt/51639908
		# for each parameter (or something like this), set if it requires gradient or not
		for param in target_layer.parameters():
			param.requires_grad = is_require_gradient

	# Interface / Virtual
	def getCodePath(self) -> str:
		return __file__
