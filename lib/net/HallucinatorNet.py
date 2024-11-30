# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from lib.net.FBNet import define_G
from lib.net.net_util import init_net, VGGLoss
from lib.net.HGFilters import *
from lib.net.BasePIFuNet import BasePIFuNet
import torch
import torch.nn as nn


class Hallucinator(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self, cfg, error_term=nn.SmoothL1Loss()):

        super(Hallucinator, self).__init__(error_term=error_term)

        self.l1_loss = nn.SmoothL1Loss()

        self.opt = cfg.net

        if self.training:
            self.vgg_loss = [VGGLoss()]

        self.in_nmlB = [
            item[0] for item in self.opt.in_nml
            if '_B' in item[0] or item[0] == 'image'
        ]
        self.in_nmlL = [
            item[0] for item in self.opt.in_nml
            if '_L' in item[0] or item[0] == 'image'
        ]
        self.in_nmlB_dim = sum([
            item[1] for item in self.opt.in_nml
            if '_B' in item[0] or item[0] == 'image'
        ])
        self.in_nmlL_dim = sum([
            item[1] for item in self.opt.in_nml
            if '_L' in item[0] or item[0] == 'image'
        ])

        self.netB = define_G(self.in_nmlB_dim, 3, 64, "global", 4, 9, 1, 3,
                             "instance")
        self.netL = define_G(self.in_nmlL_dim, 3, 64, "global", 4, 9, 1, 3,
                             "instance")

        init_net(self)

    def forward(self, in_tensor):

        inB_list = []
        inL_list = []

        for name in self.in_nmlB:
            inB_list.append(in_tensor[name])
        for name in self.in_nmlL:
            inL_list.append(in_tensor[name])

        nmlB = self.netB(torch.cat(inB_list, dim=1))
        nmlL = self.netL(torch.cat(inL_list, dim=1))

        # ||normal|| == 1
        nmlB = nmlB / torch.norm(nmlB, dim=1, keepdim=True)
        nmlL = nmlL / torch.norm(nmlL, dim=1, keepdim=True)

        # output: float_arr [-1,1] with [B, C, H, W]

        mask = (in_tensor['image'].abs().sum(dim=1, keepdim=True) !=
                0.0).detach().float()

        nmlB = nmlB * mask
        #nmlL = nmlL * mask

        return nmlB, nmlL

    def get_norm_error(self, prd_B, prd_L, tgt):
        """calculate normal loss

        Args:
            pred (torch.tensor): [B, 6, 512, 512]
            tagt (torch.tensor): [B, 6, 512, 512]
        """

        tgt_B, tgt_L = tgt['render_B'], tgt['render_L']

        l1_B_loss = self.l1_loss(prd_B, tgt_B)
        l1_L_loss = self.l1_loss(prd_L, tgt_L)

        with torch.no_grad():
            vgg_B_loss = self.vgg_loss[0](prd_B, tgt_B)
            vgg_L_loss = self.vgg_loss[0](prd_L, tgt_L)

        total_loss = [
            5.0 * l1_B_loss + vgg_B_loss, 5.0 * l1_L_loss + vgg_L_loss
        ]

        return total_loss
