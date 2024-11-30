# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.autograd import grad
# from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
import numpy as np

class SDF2Density(pl.LightningModule):
    def __init__(self):
        super(SDF2Density, self).__init__()

        # learnable parameters beta, with initial value 0.1
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, sdf):
        # use Laplace CDF to compute the probability
        # temporally use sigmoid to represent laplace CDF
        return 1.0/(self.beta+1e-6)*F.sigmoid(-sdf/(self.beta+1e-6))

class SDF2Occ(pl.LightningModule):
    def __init__(self):
        super(SDF2Occ, self).__init__()

        # learnable parameters beta, with initial value 0.1
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, sdf):
        # use Laplace CDF to compute the probability
        # temporally use sigmoid to represent laplace CDF
        return F.sigmoid(-sdf/(self.beta+1e-6))


class DeformationMLP(pl.LightningModule):
    def __init__(self,input_dim=64,output_dim=3,activation='LeakyReLU',name=None,opt=None):
        super(DeformationMLP, self).__init__()
        self.name = name
        self.activation = activation
        self.activate = nn.LeakyReLU(inplace=True)
        # self.mlp = nn.Sequential(
        #     nn.Conv1d(input_dim+8+1+3, 64, 1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(64, output_dim, 1),
        #     )
        channels=[input_dim+8+1+3,128, 64, output_dim]
        self.deform_mlp=MLP(filter_channels=channels,
                         name="if",
                         res_layers=opt.res_layers,
                         norm=opt.norm_mlp,
                         last_op=None)  # occupancy
        smplx_dim = 10475
        k=8
        self.per_pt_code = nn.Embedding(smplx_dim,k)

    def forward(self, feature,smpl_vis,pts_id, xyz):
        '''
        feature may include multiple view inputs
        args:
            feature: [B, C_in, N]
        return:
            [B, C_out, N] prediction
        '''
        y = feature
        e_code=self.per_pt_code(pts_id).permute(0,2,1)    # a code that distinguishes each point on different parts of the body
        y=torch.cat([y,xyz,smpl_vis,e_code],1)
        y = self.deform_mlp(y)
        return y

class MLP(pl.LightningModule):

    def __init__(self,
                 filter_channels,
                 name=None,
                 res_layers=[],
                 norm='group',
                 last_op=None):

        super(MLP, self).__init__()

        self.filters = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_layers = res_layers
        self.norm = norm
        self.last_op = last_op
        self.name = name
        self.activate = nn.LeakyReLU(inplace=True)

        for l in range(0, len(filter_channels) - 1):
            if l in self.res_layers:
                self.filters.append(
                    nn.Conv1d(filter_channels[l] + filter_channels[0],
                              filter_channels[l + 1], 1))
            else:
                self.filters.append(
                    nn.Conv1d(filter_channels[l], filter_channels[l + 1], 1))

            if l != len(filter_channels) - 2:
                if norm == 'group':
                    self.norms.append(nn.GroupNorm(32, filter_channels[l + 1]))
                elif norm == 'batch':
                    self.norms.append(nn.BatchNorm1d(filter_channels[l + 1]))
                elif norm == 'instance':
                    self.norms.append(nn.InstanceNorm1d(filter_channels[l +
                                                                        1]))
                elif norm == 'weight':
                    self.filters[l] = nn.utils.weight_norm(self.filters[l],
                                                           name='weight')
                    # print(self.filters[l].weight_g.size(),
                    #       self.filters[l].weight_v.size())

    def forward(self, feature):
        '''
        feature may include multiple view inputs
        args:
            feature: [B, C_in, N]
        return:
            [B, C_out, N] prediction
        '''
        y = feature
        tmpy = feature

        for i, f in enumerate(self.filters):

            y = f(y if i not in self.res_layers else torch.cat([y, tmpy], 1))
            if i != len(self.filters) - 1:
                if self.norm not in ['batch', 'group', 'instance']:
                    y = self.activate(y)
                else:
                    y = self.activate(self.norms[i](y))

        if self.last_op is not None:
            y = self.last_op(y)

        return y


# Positional encoding (section 5.1)
class Embedder(pl.LightningModule):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires=6, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Transformer encoder layer
# uses Embedder to add positional encoding to input points
# uses query points as query, deformed points as key, point features as value for attention
class TransformerEncoderLayer(pl.LightningModule):
    def __init__(self, d_model=256, skips=4, multires=6, num_mlp_layers=8, dropout=0.1, opt=None):
        super(TransformerEncoderLayer, self).__init__()

        embed_fn, input_ch = get_embedder(multires=multires)
        self.skips=skips
        self.dropout = dropout
        D=num_mlp_layers
        self.positional_encoding = embed_fn
        self.d_model = d_model
        triplane_dim=64
        opt.mlp_dim[0]=triplane_dim+6+8
        opt.mlp_dim_color[0]=triplane_dim+6+8

        self.geo_mlp=MLP(filter_channels=opt.mlp_dim,
                         name="if",
                         res_layers=opt.res_layers,
                         norm=opt.norm_mlp,
                         last_op=nn.Sigmoid())  # occupancy
        
        self.color_mlp=MLP(filter_channels=opt.mlp_dim_color,
                           name="color_if",
                           res_layers=opt.res_layers,
                           norm=opt.norm_mlp,
                           last_op=nn.Tanh())  # color

        self.softmax = nn.Softmax(dim=-1)



    def forward(self,query_points,key_points,point_features,smpl_feat,training=True,type='shape'):
        # Q=self.positional_encoding(query_points)  #[B,N,39]
        # K=self.positional_encoding(key_points)   #[B,N',39]
        # V=point_features.permute(0,2,1)                                     #[B,N',192]
        # t=0.1
        # #attn_output, attn_output_weights = self.attention(Q.permute(1,0,2), K.permute(1,0,2), V.permute(1,0,2))  #[B,N,192]
        # attn_output_weights = torch.bmm(Q, K.transpose(1, 2))  #[B,N,N']
        # attn_output_weights = self.softmax(attn_output_weights/t)  #[B,N,N']
        # # drop out
        # attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=True)
        # # master feature
        # attn_output = torch.bmm(attn_output_weights, V)  #[B,N,192]

        attn_output=point_features                       # [B,N,192] bary centric interpolation 

        feature=torch.cat([attn_output,smpl_feat],dim=1)               
       
        if type=='shape':
            h=feature          
           
            h=self.geo_mlp(h)   # [B,1,N]
            return h
        
        
        elif type=='color':
            #f=self.head(feature)               #[B,N,512]

            h=feature
           
            h=self.color_mlp(h)   # [B,3,N]
            return h
        elif type=='shape_color':
            h_s=feature
            h_c=feature
           
            h_s=self.geo_mlp(h_s)   # [B,1,N]
           
            h_c=self.color_mlp(h_c)   # [B,3,N]
            
            return h_s,h_c
            



class Swish(pl.LightningModule):
    def __init__(self):
        super(Swish, self).__init__()
 
    def forward(self, x):
        x = x * F.sigmoid(x)
        return x
    








# # Import pytorch modules
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# Define positional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# # Define model parameters
# d_model = 256 # output size of MLP
# nhead = 8 # number of attention heads
# dim_feedforward = 512 # hidden size of MLP
# num_layers = 2 # number of MLP layers
# num_frequencies = 6 # number of frequencies for positional encoding
# dropout = 0.1 # dropout rate

# # Define model components
# pos_encoder = PositionalEncoding(d_model, num_frequencies) # positional encoding layer
# encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) # transformer encoder layer
# encoder = nn.TransformerEncoder(encoder_layer, num_layers) # transformer encoder
# mlp_geo = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model)) # MLP for geometry
# mlp_alb = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model)) # MLP for albedo
# head_geo = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 3)) # geometry head
# head_alb = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 3), nn.Sigmoid()) # albedo head

# # Define input tensors
# # deformed body points: (batch_size, num_points, 3)
# x = torch.randn(batch_size, num_points, 3)
# # query point positions: (batch_size, num_queries, 3)
# y = torch.randn(batch_size, num_queries, 3)

# # Map both d


