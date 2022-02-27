# from __future__ import  absolute_import
# import os
# import time
# from collections import namedtuple
# from utils.utils import AnchorTargetCreator,ProposalTargetCreator
# from torch.nn import functional as F
# from torch import nn
# import torch as torch
# import cv2
# import numpy as np
# import array_tool as at
# import matplotlib.pyplot as plt
# from bbox_tools import loc2bbox
#
# # LossTuple = namedtuple('LossTuple',
# #                        ['rpn_loc_loss',
# #                         'rpn_cls_loss',
# #                         'roi_loc_loss',
# #                         'roi_cls_loss',
# #                         'roi_importance_loss',
# #                         'total_loss'
# #                         ])
# def _PositionalEmbedding( f_g, dim_g=128, wave_len=1000):
#     f_g = f_g.view(-1,4)
#     x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)
#
#     cx = (x_min + x_max) * 0.5
#     cy = (y_min + y_max) * 0.5
#     w = (x_max - x_min) + 1.
#     h = (y_max - y_min) + 1.
#
#     delta_x = cx - cx.view(1, -1)
#     delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
#     delta_x = torch.log(delta_x)
#
#     delta_y = cy - cy.view(1, -1)
#     delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
#     delta_y = torch.log(delta_y)
#
#     delta_w = torch.log(w / w.view(1, -1))
#     delta_h = torch.log(h / h.view(1, -1))
#     size = delta_h.size()
#
#     delta_x = delta_x.view(size[0], size[1], 1)
#     delta_y = delta_y.view(size[0], size[1], 1)
#     delta_w = delta_w.view(size[0], size[1], 1)
#     delta_h = delta_h.view(size[0], size[1], 1)
#
#     position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)
#
#     feat_range = torch.arange(dim_g / 8).cuda()
#     dim_mat = feat_range / (dim_g / 8)
#     dim_mat = 1. / (torch.pow(wave_len, dim_mat))
#
#     dim_mat = dim_mat.view(1, 1, 1, -1)
#     position_mat = position_mat.view(size[0], size[1], 4, -1)
#     position_mat = 100. * position_mat
#
#     mul_mat = position_mat * dim_mat
#     mul_mat = mul_mat.view(size[0], size[1], -1)
#     sin_mat = torch.sin(mul_mat)
#     cos_mat = torch.cos(mul_mat)
#     embedding = torch.cat((sin_mat, cos_mat), -1)
#     return embedding
#
# class FasterRCNNTrainer(nn.Module):
#     def __init__(self, faster_rcnn,optimizer):
#         super(FasterRCNNTrainer, self).__init__()
#
#         self.faster_rcnn = faster_rcnn
#         self.rpn_sigma = 3
#         self.roi_sigma = 1
#         self.n_class=2
#
#         # target creator create gt_bbox gt_label etc as training targets.
#         self.anchor_target_creator = AnchorTargetCreator()
#         self.proposal_target_creator = ProposalTargetCreator()
#
#         self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
#         self.loc_normalize_std = faster_rcnn.loc_normalize_std
#
#         self.optimizer = optimizer
#
#         self.class_score = nn.Linear(2048, self.n_class, bias=True).cuda()
#         self.sigmoid = nn.Sigmoid().cuda()
#         self.bbx_predict = nn.Linear(2048, self.n_class*4, bias=True).cuda()
#
#     def forward(self, imgs, bboxes, labels, scale):
#
#
#         n = imgs.shape[0]
#         if n != 1:
#             raise ValueError('Currently only batch size 1 is supported.')
#
#         _, _, W, H = imgs.shape
#         img_size = (W, H)
#
#         # 获取真实框和标签
#         bbox = bboxes[0]
#         label = labels[0]
#
#         # label1 = labels1[0]
#
#         # 获取公用特征层
#         features = self.faster_rcnn.extractor(imgs)#1,1024,80,45
#
#         '建议框和对应score'
#         rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features, img_size, scale)
#         #所有候选框坐标rpn_locs[32400,4] 所有候选框分类rpn_scores[32400,2] 分数最高的候选框rois[list[0:2000][1,4]] roi_indices[2000]anchor[32400,4]]
#         # 获取建议框的置信度和回归系数
#         rpn_score = rpn_scores[0]
#         rpn_loc = rpn_locs[0]
#         roi = rois
#         # ------------------------------------------ #
#         #   建议框网络的loss
#         # ------------------------------------------ #
#         # 先获取建议框网络应该有的预测结果
#         gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox.cpu().numpy(),anchor,img_size)
#         #[gt_rpn_loc[32400,4]], [gt_rpn_label[32400]]
#
#         # print(gt_rpn_label)
#         # print(sum(gt_rpn_label[gt_rpn_label==-1]))
#         gt_rpn_label = torch.Tensor(gt_rpn_label).long()
#         gt_rpn_loc = torch.Tensor(gt_rpn_loc)
#
#         # 计算建议框网络的loss值#
#         rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma)
#         # if rpn_score.is_cuda:
#         gt_rpn_label = gt_rpn_label.cuda()
#
#         rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)
#
#         #[128,1024,14,14]
#         # ------------------------------------------ #
#         #   classifier网络的loss
#         # ------------------------------------------ #
#         sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox.cpu().numpy(), label.cpu().numpy(), self.loc_normalize_mean, self.loc_normalize_std)
#         # print(gt_roi_loc)
#         #sample_roi[list [1:128][1,4]] gt_roi_loc[list [1:128][1,4]] gt_roi_label[0:128] roi[list[0:2000][1,4]] bbox[13,4] label[13,2]
#         # sample_roi, gt_roi_loc, gt_roi_label1 = self.proposal_target_creator(roi, bbox.cpu().numpy(),label1.cpu().numpy(),self.loc_normalize_mean,self.loc_normalize_std)
#         sample_roi_index = torch.zeros(len(sample_roi))
#
#         roi_cls_loc, roi_score = self.faster_rcnn.head(features,sample_roi,sample_roi_index)#features[1,1024,80,45]
#
#
#         #roi_cls_loc[128,12] roi_score[128,3]
#
#         # roi = at.totensor(sample_roi)#[128,4]
#
#         #[128,2048]
#         ##for duplicate removal
#         # mean = torch.Tensor(self.loc_normalize_mean).cuda().repeat(self.n_class)[None]
#         # std = torch.Tensor(self.loc_normalize_std).cuda().repeat(self.n_class)[None]
#         # roi_cls_loc = (roi_cls_loc * std + mean)
#         # roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)#[128,3,4]
#         # roi = roi.view(-1,1, 4).expand_as(roi_cls_loc)#[128,3,4]
#         # cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),at.tonumpy(roi_cls_loc).reshape((-1, 4)))#[0:384][1,4]
#         # cls_bbox = at.totensor(cls_bbox)
#         # cls_bbox = cls_bbox.view(-1, self.n_class, 4)#[128,3,4]
#         # # clip bounding box
#         # cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=img_size[0])
#         # cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=img_size[1])
#         # prob = F.softmax(at.tovariable(roi_score), dim=1) #[128,3]
#         # prob, prob_argmax = torch.max(prob, dim=-1)#[128,3] [128]
#         # cls_bbox = cls_bbox[np.arange(start=0, stop=128), prob_argmax]#[128,4]
#         # nonzero_idx = torch.nonzero(prob_argmax) #[36,1]
#         # nonzero_idx = nonzero_idx[:, 0]
#         # prob = prob[nonzero_idx]#[36]
#         # cls_bbox = cls_bbox[nonzero_idx]#[36,4]
#         # appearance_features_nobg = self.ROI_feature[nonzero_idx]#[36,2048]
#         # sorted_score, prob_argsort = torch.sort(prob, descending=True) #[36] [36]
#         # # sorted_prob = prob[prob_argsort]
#         # sorted_cls_bboxes = cls_bbox[prob_argsort] #[36,4]
#         # sorted_features = appearance_features_nobg[prob_argsort] #[36,2048]
#
#
#         n_sample = roi_cls_loc.shape[0]
#         roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)#[128,3,4] num_of_cloass+1=3
#         roi_score = roi_score.view(n_sample, -1)#[128,3,4] num_of_cloass+1=3
#
#
#         # if roi_cls_loc.is_cuda:
#         roi_loc = roi_cls_loc[torch.arange(0, n_sample).long(), torch.Tensor(gt_roi_label).long()].cuda()
#         #[128,4]
#
#
#
#         # else:
#             # roi_loc = roi_cls_loc[torch.arange(0, n_sample).long(), torch.Tensor(gt_roi_label).long()]
#         # roi_cls_loc=roi_cls_loc.cuda()
#
#         # position_embedding = _PositionalEmbedding(roi_cls_loc, 128) #[36,36,128]
#         # relation_feature = self.relation_module(self.ROI_feature,position_embedding)
#
#         #relation_feature [1, 34, 2048]
#         # score, bbx = self._region_classification(relation_feature)
#         #score[38] bbx[38,12]
#         # bbx = bbx.view(n_sample, -1, 4)  # [128,3,4] num_of_cloass+1=3
#         # if roi_cls_loc.is_cuda:
#         # bbx = bbx[torch.arange(0, n_sample).long(), torch.Tensor(gt_roi_label).long()].cuda()
#
#
#
#
#         gt_roi_label = torch.Tensor(gt_roi_label).long()
#         # gt_roi_label1 = torch.Tensor(gt_roi_label1).long()
#         gt_roi_loc = torch.Tensor(gt_roi_loc)
#         # print(roi_loc[0],'   ',gt_roi_loc[0])
#         roi_loc_loss = _fast_rcnn_loc_loss(roi_loc.contiguous(),gt_roi_loc,gt_roi_label.data, self.roi_sigma)
#                                             #roi_loc [128,4]  gt_roi_loc[128,4]  gt_roi_label[128]
#         # if roi_score.is_cuda:
#         gt_roi_label = gt_roi_label.cuda()
#         # gt_roi_label1 = gt_roi_label1.cuda()
#
#
#         roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)
#
#         print(roi_score,gt_roi_label)
#
#
#         # print(importance.shape, gt_roi_label[:,1].shape)
#
#         # print(importance,gt_roi_label[:,1])
#         # importance_loss = nn.CrossEntropyLoss()(importance, gt_roi_label[:,1])
#
#         # print(importance_loss)
#         # roi_importance_loss = nn.CrossEntropyLoss()(roi_importance, gt_roi_label1)#[][128]
#         roi_cls_loss=roi_cls_loss
#         losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
#         # losses = losses + [sum(losses)]
#         return (losses)
#
#     def train_step(self, imgs, bboxes, labels, scale):
#         self.optimizer.zero_grad()
#         losses = self.forward(imgs, bboxes, labels, scale)
#         sum(losses).backward()
#         self.optimizer.step()
#         return losses
#
#     def _region_classification(self, relational_feature):  # dection reg and classify head
#
#         pred_scores = self.class_score(relational_feature)
#         pred_prob = self.sigmoid(pred_scores)
#         pred_bbx = self.bbx_predict(relational_feature)
#
#         return pred_prob, pred_bbx
#
# def _smooth_l1_loss(x, t, in_weight, sigma):
#     sigma2 = sigma ** 2
#     diff = in_weight * (x - t)
#     abs_diff = diff.abs()
#     flag = (abs_diff.data < (1. / sigma2)).float()
#     y = (flag * (sigma2 / 2.) * (diff ** 2) +
#          (1 - flag) * (abs_diff - 0.5 / sigma2))
#     return y.sum()
#
# def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
#     # print(pred_loc.shape,gt_loc.shape,gt_label.shape)
#
#     in_weight = torch.zeros(gt_loc.shape)
#     in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
#
#     # if pred_loc.is_cuda:
#
#     gt_loc = gt_loc.cuda()
#     in_weight = in_weight.cuda()
#     # smooth_l1损失函数
#     loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
#     # 进行标准化
#     loc_loss /= ((gt_label >= 0).sum().float())
#     return loc_loss
#
#
#
#
#
#
#
# class RelationModule(nn.Module):
#     def __init__(self,n_relations = 16, appearance_feature_dim=1024,key_feature_dim = 128, geo_feature_dim = 128, isDuplication = False):
#         super(RelationModule, self).__init__()
#         self.isDuplication=isDuplication#false
#         self.Nr = n_relations#16
#         self.dim_g = geo_feature_dim#64
#         self.relation = nn.ModuleList()#16
#         for N in range(self.Nr):
#             self.relation.append(RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim))#1024,64,64
#     def forward(self, f_a, position_embedding ):
#         # f_a, position_embedding = input_data
#         isFirst=True
#         for N in range(self.Nr):
#             # print(self.relation[N])
#             if(isFirst):
#                 concat = self.relation[N](f_a,position_embedding)
#                 isFirst=False
#             else:
#                 concat = torch.cat((concat, self.relation[N](f_a, position_embedding)), -1)
#
#         return concat+f_a
#
#         # return f_a
#
# class RelationUnit(nn.Module):
#     def __init__(self, appearance_feature_dim=1024,key_feature_dim = 128, geo_feature_dim = 128):
#         super(RelationUnit, self).__init__()
#         self.dim_g = geo_feature_dim
#         self.dim_k = key_feature_dim
#         self.WG = nn.Linear(geo_feature_dim, 1, bias=True)
#         self.WK = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
#         self.WQ = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
#         self.WV = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
#         self.fc = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
#
#         self.relu = nn.ReLU(inplace=True)
#
#
#     def forward(self, f_a, position_embedding):
#
#         N=f_a.size()[0]
#         # N= f_a.size()
#         #f_a[128,1024,14,14]
#         position_embedding = position_embedding.view(-1,self.dim_g)
#         #position_embedding [29,29,128]----[841,128]
#         w_g = self.relu(self.WG(position_embedding))
#         # #[16384,1]
#         f_a = f_a.view(128,-1)
#         w_k = self.WK(f_a)
#         #w_k[128, 128]
#         w_k = w_k.view(N,1,self.dim_k)
#
#         w_q = self.WQ(f_a)
#         w_q = w_q.view(1,N,self.dim_k)
#
#         scaled_dot = torch.sum((w_k*w_q),-1 )
#         scaled_dot = scaled_dot / np.sqrt(self.dim_k)
#
#         w_g = w_g.view(N*3,N*3)
#         w_a = scaled_dot.view(N,N)
#
#         w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + w_a
#         w_mn = torch.nn.Softmax(dim=1)(w_mn)
#
#         w_v = self.WV(f_a)
#
#         w_mn = w_mn.view(N,N,1)
#         w_v = w_v.view(N,1,-1)
#
#         output = w_mn*w_v
#
#         output = torch.sum(output,-2)
#         return output
#


from __future__ import absolute_import

import os
import time
from collections import namedtuple

import torch as torch
from torch import nn
from torch.nn import functional as F

from utils.utils import AnchorTargetCreator, ProposalTargetCreator

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn, optimizer):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = 1
        self.roi_sigma = 1

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = [0, 0, 0, 0]
        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]

        self.optimizer = optimizer

    def forward(self, imgs, bboxes, labels, scale):
        n = imgs.shape[0]
        img_size = imgs.shape[2:]

        # 获取公用特征层
        base_feature = self.faster_rcnn.extractor(imgs)

        # 利用rpn网络获得先验框的得分与调整参数
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(base_feature, img_size, scale)

        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all = 0, 0, 0, 0
        for i in range(n):
            bbox = bboxes[i]
            label = labels[i]
            rpn_loc = rpn_locs[i]
            rpn_score = rpn_scores[i]
            roi = rois[roi_indices == i]
            feature = base_feature[i]

            # -------------------------------------------------- #
            #   利用真实框和先验框获得建议框网络应该有的预测结果
            #   给每个先验框都打上标签
            #   gt_rpn_loc      [num_anchors, 4]
            #   gt_rpn_label    [num_anchors, ]
            # -------------------------------------------------- #
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor, img_size)
            gt_rpn_loc = torch.Tensor(gt_rpn_loc)
            gt_rpn_label = torch.Tensor(gt_rpn_label).long()

            if rpn_loc.is_cuda:
                gt_rpn_loc = gt_rpn_loc.cuda()
                gt_rpn_label = gt_rpn_label.cuda()

            # -------------------------------------------------- #
            #   分别计算建议框网络的回归损失和分类损失
            # -------------------------------------------------- #
            rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

            # ------------------------------------------------------ #
            #   利用真实框和建议框获得classifier网络应该有的预测结果
            #   获得三个变量，分别是sample_roi, gt_roi_loc, gt_roi_label
            #   sample_roi      [n_sample, ]
            #   gt_roi_loc      [n_sample, 4]
            #   gt_roi_label    [n_sample, ]
            # ------------------------------------------------------ #
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label,
                                                                                self.loc_normalize_mean,
                                                                                self.loc_normalize_std)
            sample_roi = torch.Tensor(sample_roi)
            gt_roi_loc = torch.Tensor(gt_roi_loc)
            gt_roi_label = torch.Tensor(gt_roi_label).long()

            sample_roi_index = torch.zeros(len(sample_roi))

            if feature.is_cuda:
                sample_roi = sample_roi.cuda()
                sample_roi_index = sample_roi_index.cuda()
                gt_roi_loc = gt_roi_loc.cuda()
                gt_roi_label = gt_roi_label.cuda()

            roi_cls_loc, roi_score = self.faster_rcnn.head(torch.unsqueeze(feature, 0), sample_roi, sample_roi_index, img_size)

            # ------------------------------------------------------ #
            #   根据建议框的种类，取出对应的回归预测结果
            # ------------------------------------------------------ #
            n_sample = roi_cls_loc.size()[1]
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]

            # -------------------------------------------------- #
            #   分别计算Classifier网络的回归损失和分类损失
            # -------------------------------------------------- #
            roi_loc_loss = _fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score[0], gt_roi_label)
            # print(roi_score[0],gt_roi_label)
            # print(roi_loc[0], gt_roi_loc[0])
            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss
            roi_loc_loss_all += roi_loc_loss
            roi_cls_loss_all += roi_cls_loss

        losses = [rpn_loc_loss_all / n, rpn_cls_loss_all / n, roi_loc_loss_all / n, roi_cls_loss_all / n]
        losses = losses + [sum(losses)]
        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses


def _smooth_l1_loss(x, t, sigma):
    sigma_squared = sigma ** 2
    regression_diff = (x - t)
    regression_diff = regression_diff.abs()
    regression_loss = torch.where(
        regression_diff < (1. / sigma_squared),
        0.5 * sigma_squared * regression_diff ** 2,
        regression_diff - 0.5 / sigma_squared
    )
    return regression_loss.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    pred_loc = pred_loc[gt_label > 0]
    gt_loc = gt_loc[gt_label > 0]

    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, sigma)
    num_pos = (gt_label > 0).sum().float()
    loc_loss /= torch.max(num_pos, torch.ones_like(num_pos))
    return loc_loss











