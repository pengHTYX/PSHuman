import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
import functools

from .net_util import *
from lib.dataset.PointFeat import PointFeat
from lib.dataset.mesh_util import feat_select


# class ResBlkPIFuNet(BasePIFuNet):
#     def __init__(self, opt,
#                  projection_mode='orthogonal'):
#         if opt.color_loss_type == 'l1':
#             error_term = nn.L1Loss()
#         elif opt.color_loss_type == 'mse':
#             error_term = nn.MSELoss()

#         super(ResBlkPIFuNet, self).__init__(
#             projection_mode=projection_mode,
#             error_term=error_term)

#         self.name = 'respifu'
#         self.opt = opt
#         self.smpl_feats = self.opt.smpl_feats
#         norm_type = get_norm_layer(norm_type=opt.norm_color)
#         self.image_filter = ResnetFilter(opt, norm_layer=norm_type)
#         self.smpl_feat_dict=None

#         self.surface_classifier = SurfaceClassifier(
#             filter_channels=self.opt.mlp_dim_color,
#             num_views=self.opt.num_views,
#             no_residual=self.opt.no_residual,
#             last_op=nn.Tanh())

#         self.normalizer = DepthNormalizer(opt)

#         init_net(self)

#     def filter(self, images):
#         '''
#         Filter the input images
#         store all intermediate features.
#         :param images: [B, C, H, W] input images
#         '''
#         self.im_feat = self.image_filter(images)

#     def attach(self, im_feat):
#         #self.im_feat = torch.cat([im_feat, self.im_feat], 1)
#         self.geo_feat=im_feat

#     def query(self, points, calibs, transforms=None, labels=None):
#         '''
#         Given 3D points, query the network predictions for each point.
#         Image features should be pre-computed before this call.
#         store all intermediate features.
#         query() function may behave differently during training/testing.
#         :param points: [B, 3, N] world space coordinates of points
#         :param calibs: [B, 3, 4] calibration matrices for each image
#         :param transforms: Optional [B, 2, 3] image space coordinate transforms
#         :param labels: Optional [B, Res, N] gt labeling
#         :return: [B, Res, N] predictions for each point
#         '''
#         if labels is not None:
#             self.labels = labels

        
#         xyz = self.projection(points, calibs, transforms)
#         xy = xyz[:, :2, :]
#         z = xyz[:, 2:3, :]

#         z_feat = self.normalizer(z)


#         if self.smpl_feat_dict==None:
#             # This is a list of [B, Feat_i, N] features
#             point_local_feat_list = [self.index(self.im_feat, xy), z_feat]
#             # [B, Feat_all, N]
#             point_local_feat = torch.cat(point_local_feat_list, 1)

#             self.preds = self.surface_classifier(point_local_feat)
#         else:
#             point_feat_extractor = PointFeat(self.smpl_feat_dict["smpl_verts"],
#                                              self.smpl_feat_dict["smpl_faces"])
#             point_feat_out = point_feat_extractor.query(
#                 xyz.permute(0, 2, 1).contiguous(), self.smpl_feat_dict)
            
#             feat_lst = [
#                 point_feat_out[key] for key in self.smpl_feats
#                 if key in point_feat_out.keys()
#             ]
#             smpl_feat = torch.cat(feat_lst, dim=2).permute(0, 2, 1)
#             point_normal_feat = feat_select(self.index(self.geo_feat, xy),   # select front or back normal feature
#                                                    smpl_feat[:, [-1], :])
#             point_color_feat = torch.cat([self.index(self.im_feat, xy), z_feat],1)
#             point_feat_list = [point_normal_feat, point_color_feat, smpl_feat[:, :-1, :]]
#             point_feat = torch.cat(point_feat_list, 1)
#             self.preds = self.surface_classifier(point_feat)

#     def forward(self, images, im_feat, points, calibs, transforms=None, labels=None):
        
#         self.filter(images)

#         self.attach(im_feat)
        

#         self.query(points, calibs, transforms, labels)

        
#         error = self.get_error(self.preds,self.labels)

#         return self.preds, error

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, last=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, last)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, last=False):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if last:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetFilter(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, opt, input_nc=3, output_nc=256, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetFilter, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            if i == n_blocks - 1:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias, last=True)]
            else:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias)]

        if opt.use_tanh:
            model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
