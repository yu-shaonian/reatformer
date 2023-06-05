import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *

import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np
from einops import repeat

from einops.layers.torch import Rearrange
from einops import rearrange

from .core.utils.utils import coords_grid, bilinear_sampler, upflow8
from .core.FlowFormer.common import FeedForward, pyramid_retrieve_tokens, sampler, sampler_gaussian_fix, retrieve_tokens, MultiHeadAttention, MLP
from .core.FlowFormer.encoders import twins_svt_large_context, twins_svt_large
from .core.position_encoding import PositionEncodingSine, LinearPositionEncoding
from .core.FlowFormer.LatentCostFormer.twins import PosConv
from .core.FlowFormer.LatentCostFormer.encoder import MemoryEncoder,CostPerceiverEncoder
from .core.FlowFormer.LatentCostFormer.decoder import MemoryDecoder
from .core.FlowFormer.LatentCostFormer.cnn import BasicEncoder

from models.core.default import get_cfg
from models.mvsformer import MvsformerEncoder,MvsformerDecoder



class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)


    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined



class MVSNet(nn.Module):
    def __init__(self, refine=True):
        super(MVSNet, self).__init__()
        self.refine = refine

        self.feature = FeatureNet(base_channels=3)
        self.feat_encoder = BasicEncoder(input_dim=3,output_dim=32, norm_fn='instance')
        self.context_encoder = BasicEncoder(input_dim=3,output_dim=20, norm_fn='instance')
        self.cost_regularization = CostRegNet()
        # 这里的Conv2d卷积包括了norm和relu,自定义卷积
        self.MvsformerEncoder = MvsformerEncoder()
        self.MvsformerDecoder = MvsformerDecoder()
        if self.refine:
            self.refine_network = RefineNet()

        cfg = get_cfg()

        self.memory_decoder = MemoryDecoder(cfg['latentcostformer'])
        self.cost_perceiver_encoder = CostPerceiverEncoder(cfg['latentcostformer'])

    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1)
        B, _, H, W = imgs[0].shape


        # WARNING!! tmp, to reduce GPU memory
        # imgs = [F.interpolate(img, scale_factor=0.5) for img in imgs]
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        num_depth = depth_values.shape[1]

        # step 1. feature extraction
        # in: images; out: 20-channel feature maps
        features = [self.feature(img) for img in imgs]
        # 这里使用transmvs提取特征的方法
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # context = self.context_encoder(imgs[0])，
        # 通过卷积的方式实现下采样
        # context = self.downsample3(imgs[0])
        # context = self.downsample4(context)
        # print('context开始1：',context.shape)
        # 这里单独提权ref图像的特征用于后面信息交融

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        similarity_sum = 0
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            src_proj_new = src_proj[:, 0].clone() # [B, 4, 4]
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone() # [B, 4, 4]
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
            similarity = (warped_volume * ref_volume)
            similarity_sum = similarity_sum + similarity

        cost_volume = similarity_sum / len(src_projs)
        # [B, C , C, H, W]
        # 将cost_volume尺寸缩小，并且通道数减少，
        # context [B,C,H,W]
        # import ipdb
        # ipdb.set_trace()

        del imgs,  proj_matrices,ref_volume, features
        data = {}
        # cost_volume_ = self.cost_perceiver_encoder(cost_volume, data, context)
        del similarity, similarity_sum, warped_volume
        B,C1,C2,H,W = cost_volume.shape
        cost_volume = cost_volume.reshape(B,C1*C2,H,W)
        depth = self.MvsformerEncoder(cost_volume)
        del cost_volume
        depth = self.MvsformerDecoder(depth)

        prob_volume = torch.exp(F.log_softmax(depth, dim=1))
        depth = depth_wta(prob_volume, depth_values=depth_values)
        with torch.no_grad():
            photometric_confidence = torch.max(prob_volume, dim=1)[0]
        return {"depth": depth,  "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values}
        # import ipdb
        # ipdb.set_trace()
        # depth_predictions_all = []
        # depth_value = depth_values.squeeze(0).flatten()
        # M = depth_values.shape[1]
        # bidx = (torch.arange(B) * M).to(device=depth_value.device)
        # bidx = repeat(bidx, 'b -> b h w', h=int(H/1), w=int(W/1))
        # for i in range(len(index_predictions_all)):
        #     index_predictions_all[i] = F.interpolate(index_predictions_all[i], size=(int(H/1),int(W/1))).squeeze(1)
        #     index_pred = index_predictions_all[i]
        #     mask_left = index_pred < 0
        #     mask_right = index_pred >= M-1
        #     mask_inside = torch.logical_and(index_pred >= 0, index_pred < M-1)
        #     depth_pred = torch.zeros_like(index_pred)
        #     inside_index = torch.clip(index_pred[mask_inside], 0, M-2) + bidx[mask_inside]
        #     left_index = torch.floor(inside_index).long()
        #     right_index = left_index + 1
        #     delta = inside_index - torch.floor(inside_index)
        #
        #     depth_pred[mask_inside] = depth_value[left_index] * (1 - delta) + depth_value[right_index] * delta
        #     depth_pred[mask_left] = depth_value[bidx[mask_left]]
        #     depth_pred[mask_right] = depth_value[bidx[mask_right] + M-1]
        #     depth_predictions_all.append(depth_pred.squeeze(1))

        # return depth_predictions_all, index_predictions_all


def depth2index(depth_gt, depth_values):
    _, M = depth_values.shape
    B, H, W = depth_gt.shape
    depth_values = repeat(depth_values, 'b m -> b h w m', h=H, w=W)
    index_gt = torch.zeros_like(depth_gt)
    mask_left = depth_gt < depth_values[...,0]
    mask_right = depth_gt >= depth_values[...,-1]
    mask_inside = torch.logical_and(depth_gt >= depth_values[...,0], depth_gt < depth_values[...,-1])
    smaller_mask = depth_gt[...,None] < depth_values
    left_index = torch.clip(torch.argmax(smaller_mask.int(), dim=-1) - 1, 0, M-2)
    right_index = left_index + 1
    depth_left = torch.gather(depth_values, -1, left_index[...,None])[...,0]
    depth_right = torch.gather(depth_values, -1, right_index[...,None])[...,0]
    delta = (depth_gt - depth_left) / (depth_right - depth_left)
    index_gt[mask_left] = 0
    index_gt[mask_right] = M-1
    index_gt[mask_inside] = left_index[mask_inside] + delta[mask_inside]
    return index_gt


# def mvsnet_loss(depth_est, depth_gt, mask, depth_values, gamma=0.8):
#     # depth_preds -> List[torch]
#     """ Loss function defined over sequence of depth predictions """
#
#     # index_gt = depth2index(depth_gt, depth_values)
#     index_gt = depth_gt
#
#     n_predictions = len(depth_est)
#     mask = mask.bool()
#     depth_loss = 0.0
#     for i in range(n_predictions):
#         i_weight = gamma**(n_predictions - i - 1)
#         depth_pred = depth_est[i]
#         # i_loss = F.smooth_l1_loss(depth_pred[mask], depth_gt[mask], size_average=True)
#         i_loss = F.smooth_l1_loss(depth_pred[mask], index_gt[mask], reduce='mean')
#         depth_loss += i_weight * i_loss
#     return depth_loss

def entropy_loss(prob_volume, depth_gt, mask, depth_value, return_prob_map=False):
    # from AA
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    shape = depth_gt.shape          # B,H,W

    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)     # B,N,H,W
    else:
        depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W

    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)

    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1) # B, 1, H, W

    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map


def mvsnet_loss(inputs, depth_gt, mask, **kwargs):
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask.device, requires_grad=False)
    total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask.device, requires_grad=False)


    prob_volume = inputs["prob_volume"]
    depth_values = inputs["depth_values"]
    depth_gt = depth_gt
    mask = mask
    mask = mask > 0.5
    entropy_weight = 2.0

    entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_values)
    entro_loss = entro_loss * entropy_weight
    depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
    total_entropy += entro_loss
    total_loss += entro_loss

    return total_loss, depth_loss, total_entropy, depth_entropy