# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from ViT-Pytorch (https://github.com/lucidrains/vit-pytorch)
# Copyright (c) 2020 Phil Wang. All Rights Reserved.
# ------------------------------------------------------------------------------------

import math
import numpy as np
from typing import Union, Tuple, List, Optional
from functools import partial
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int or (int, int) of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = (grid_size, grid_size) if type(grid_size) != tuple else grid_size
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def init_weights(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
    
class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.norm = nn.LayerNorm(dim)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
        self.multi_head_attention=PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head))


    def forward(self, x: torch.FloatTensor, q_x:torch.FloatTensor) -> torch.FloatTensor:
        
        q_in = self.multi_head_attention(q_x)+q_x
        q_in = self.norm(q_in)

        q = rearrange(self.to_q(q_in),'b n (h d) -> b h n d', h = self.heads)       
        kv = self.to_kv(x).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out),q_in


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                                   PreNorm(dim, FeedForward(dim, mlp_dim))])
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class CrossTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([CrossAttention(dim, heads=heads, dim_head=dim_head),
                                   PreNorm(dim, FeedForward(dim, mlp_dim))])
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.FloatTensor, q_x:torch.FloatTensor) -> torch.FloatTensor:
        encoder_output=x
        for attn, ff in self.layers:
            x,q_in = attn(encoder_output, q_x)
            x = x + q_in
            x = ff(x) + x
            q_x=x

        return self.norm(q_x)

class ViTEncoder(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, channels: int = 3, dim_head: int = 64) -> None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) \
                                    else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        en_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.en_pos_embedding = nn.Parameter(torch.from_numpy(en_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.apply(init_weights)

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        x = self.to_patch_embedding(img)
        x = x + self.en_pos_embedding
        x = self.transformer(x)

        return x


class ViTDecoder(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, channels: int = 32, dim_head: int = 64) -> None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) \
                                    else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        de_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.de_pos_embedding = nn.Parameter(torch.from_numpy(de_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.to_pixel = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),
            nn.ConvTranspose2d(dim, channels, kernel_size=4, stride=4)
        )

        self.apply(init_weights)

    def forward(self, token: torch.FloatTensor) -> torch.FloatTensor:
        x = token + self.de_pos_embedding
        x = self.transformer(x)
        x = self.to_pixel(x)

        return x

    def get_last_layer(self) -> nn.Parameter:
        return self.to_pixel[-1].weight


class CrossAttDecoder(nn.Module):
    def __init__(self, image_size: Union[Tuple[int, int], int], patch_size: Union[Tuple[int, int], int],
                 dim: int, depth: int, heads: int, mlp_dim: int, channels: int = 32, dim_head: int = 64) -> None:
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) \
                                    else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) \
                                    else (patch_size, patch_size)


        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        de_pos_embedding = get_2d_sincos_pos_embed(dim, (image_height // patch_height, image_width // patch_width))

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width

        self.transformer = CrossTransformer(dim, depth, heads, dim_head, mlp_dim)
        
        self.de_pos_embedding = nn.Parameter(torch.from_numpy(de_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.to_pixel = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),
            nn.ConvTranspose2d(dim, channels, kernel_size=4, stride=4)
        )

        self.apply(init_weights)

    def forward(self, token: torch.FloatTensor, query_img:torch.FloatTensor) -> torch.FloatTensor:
        # batch_size=token.shape[0]
        # query=self.query.repeat(batch_size,1,1)+self.de_pos_embedding
        query=self.to_patch_embedding(query_img)+self.de_pos_embedding
        x = token + self.de_pos_embedding
        x = self.transformer(x,query)
        x = self.to_pixel(x)

        return x

    def get_last_layer(self) -> nn.Parameter:
        return self.to_pixel[-1].weight


class BaseQuantizer(nn.Module):
    def __init__(self, embed_dim: int, n_embed: int, straight_through: bool = True, use_norm: bool = True,
                 use_residual: bool = False, num_quantizers: Optional[int] = None) -> None:
        super().__init__()
        self.straight_through = straight_through
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x

        self.use_residual = use_residual
        self.num_quantizers = num_quantizers

        self.embed_dim = embed_dim
        self.n_embed = n_embed

        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        self.embedding.weight.data.normal_()
        
    def quantize(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        pass
    
    def forward(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        if not self.use_residual:
            z_q, loss, encoding_indices = self.quantize(z)
        else:
            z_q = torch.zeros_like(z)
            residual = z.detach().clone()

            losses = []
            encoding_indices = []

            for _ in range(self.num_quantizers):
                z_qi, loss, indices = self.quantize(residual.clone())
                residual.sub_(z_qi)
                z_q.add_(z_qi)

                encoding_indices.append(indices)
                losses.append(loss)

            losses, encoding_indices = map(partial(torch.stack, dim = -1), (losses, encoding_indices))
            loss = losses.mean()

        # preserve gradients with straight-through estimator
        if self.straight_through:
            z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices


class VectorQuantizer(BaseQuantizer):
    def __init__(self, embed_dim: int, n_embed: int, beta: float = 0.25, use_norm: bool = True,
                 use_residual: bool = False, num_quantizers: Optional[int] = None, **kwargs) -> None:
        super().__init__(embed_dim, n_embed, True,
                         use_norm, use_residual, num_quantizers)
        
        self.beta = beta

    def quantize(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        z_reshaped_norm = self.norm(z.view(-1, self.embed_dim))
        embedding_norm = self.norm(self.embedding.weight)
        
        d = torch.sum(z_reshaped_norm ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_norm ** 2, dim=1) - 2 * \
            torch.einsum('b d, n d -> b n', z_reshaped_norm, embedding_norm)

        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encoding_indices = encoding_indices.view(*z.shape[:-1])
        
        z_q = self.embedding(encoding_indices).view(z.shape)
        z_qnorm, z_norm = self.norm(z_q), self.norm(z)
        
        # compute loss for embedding
        loss = self.beta * torch.mean((z_qnorm.detach() - z_norm)**2) +  \
               torch.mean((z_qnorm - z_norm.detach())**2)

        return z_qnorm, loss, encoding_indices


class ViTVQ(pl.LightningModule):
    def __init__(self,image_size=512, patch_size=16,channels=3) -> None:
        super().__init__()
        
        self.encoder = ViTEncoder(image_size=image_size, patch_size=patch_size, dim=256,depth=8,heads=8,mlp_dim=2048,channels=channels)
        self.F_decoder = ViTDecoder(image_size=image_size, patch_size=patch_size, dim=256,depth=3,heads=8,mlp_dim=2048)
        self.B_decoder= CrossAttDecoder(image_size=image_size, patch_size=patch_size, dim=256,depth=3,heads=8,mlp_dim=2048)
        self.R_decoder= CrossAttDecoder(image_size=image_size, patch_size=patch_size, dim=256,depth=3,heads=8,mlp_dim=2048)
        self.L_decoder= CrossAttDecoder(image_size=image_size, patch_size=patch_size, dim=256,depth=3,heads=8,mlp_dim=2048)
        # self.quantizer = VectorQuantizer(embed_dim=32,n_embed=8192)
        # self.pre_quant = nn.Linear(512, 32)
        # self.post_quant = nn.Linear(32, 512)


    def forward(self, x: torch.FloatTensor,smpl_normal) -> torch.FloatTensor:    
        enc_out = self.encode(x)
        dec = self.decode(enc_out,smpl_normal)
        
        return dec

        
    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        h = self.encoder(x)
        # h = self.pre_quant(h)
        # quant, emb_loss, _ = self.quantizer(h)
        
        return h #, emb_loss

    def decode(self, enc_out: torch.FloatTensor,smpl_normal) -> torch.FloatTensor:
        back_query=smpl_normal['T_normal_B']
        right_query=smpl_normal['T_normal_R']
        left_query=smpl_normal['T_normal_L']
        # quant = self.post_quant(quant)
        dec_F = self.F_decoder(enc_out)
        dec_B = self.B_decoder(enc_out,back_query)
        dec_R = self.R_decoder(enc_out,right_query)
        dec_L = self.L_decoder(enc_out,left_query)
        
        return (dec_F,dec_B,dec_R,dec_L)

    # def encode_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
    #     h = self.encoder(x)
    #     h = self.pre_quant(h)
    #     _, _, codes = self.quantizer(h)
        
    #     return codes

    # def decode_codes(self, code: torch.LongTensor) -> torch.FloatTensor:
    #     quant = self.quantizer.embedding(code)
    #     quant = self.quantizer.norm(quant)
        
    #     if self.quantizer.use_residual:
    #         quant = quant.sum(-2)  
            
    #     dec = self.decode(quant)
        
    #     return dec