from typing import Any, Dict, Optional

import torch
from torch import nn



from diffusers.models.attention import  Attention
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
import math

import torch.nn.functional as F
if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class RowwiseMVAttention(Attention):
    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, *args, **kwargs
    ):
        processor = XFormersMVAttnProcessor()
        self.set_processor(processor)
        # print("using xformers attention processor")

class IPCDAttention(Attention):
    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, *args, **kwargs
    ):
        processor = XFormersIPCDAttnProcessor()
        self.set_processor(processor)
        # print("using xformers attention processor")



class XFormersMVAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        num_views=1,
        multiview_attention=True,
        cd_attention_mid=False
    ):
        # print(num_views)
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        height = int(math.sqrt(sequence_length)) 
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # from yuancheng; here attention_mask is None
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            print('Warning: using group norm, pay attention to use it in row-wise attention')
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key_raw = attn.to_k(encoder_hidden_states)
        value_raw = attn.to_v(encoder_hidden_states)

        # print('query', query.shape, 'key', key.shape, 'value', value.shape)
        # pdb.set_trace()
        def transpose(tensor):
            tensor = rearrange(tensor, "(b v) (h w) c -> b v h w c", v=num_views, h=height)
            tensor_0, tensor_1 = torch.chunk(tensor, dim=0, chunks=2)  # b v h w c
            tensor = torch.cat([tensor_0, tensor_1], dim=3)  # b v h 2w c
            tensor = rearrange(tensor, "b v h w c -> (b h) (v w) c", v=num_views, h=height)
            return tensor
        # print(mvcd_attention)
        # import pdb;pdb.set_trace()
        if cd_attention_mid:
            key = transpose(key_raw)
            value = transpose(value_raw)
            query = transpose(query)
        else:
            key = rearrange(key_raw, "(b v) (h w) c -> (b h) (v w) c", v=num_views, h=height) 
            value = rearrange(value_raw, "(b v) (h w) c -> (b h) (v w) c", v=num_views, h=height)
            query = rearrange(query, "(b v) (h w) c -> (b h) (v w) c", v=num_views, h=height) # torch.Size([192, 384, 320])


        query = attn.head_to_batch_dim(query) # torch.Size([960, 384, 64])
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if cd_attention_mid:
            hidden_states = rearrange(hidden_states, "(b h) (v w) c -> b v h w c", v=num_views, h=height)
            hidden_states_0, hidden_states_1 = torch.chunk(hidden_states, dim=3, chunks=2)  # b v h w c
            hidden_states = torch.cat([hidden_states_0, hidden_states_1], dim=0)  # 2b v h w c
            hidden_states = rearrange(hidden_states, "b v h w c -> (b v) (h w) c", v=num_views, h=height) 
        else:
            hidden_states = rearrange(hidden_states, "(b h) (v w) c -> (b v) (h w) c", v=num_views, h=height)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


class XFormersIPCDAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """
    
    def process(self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        num_tasks=2,
        num_views=6):
        ### TODO: num_views
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        height = int(math.sqrt(sequence_length)) 
        height_st = height // 3
        height_end = height - height_st
        
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # from yuancheng; here attention_mask is None
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        assert num_tasks == 2  # only support two tasks now
        
        
        # ip attn
        # hidden_states = rearrange(hidden_states, '(b v) l c -> b v l c', v=num_views)
        # body_hidden_states, face_hidden_states = rearrange(hidden_states[:, :-1, :, :], 'b v l c -> (b v) l c'), hidden_states[:, -1, :, :]
        # print(body_hidden_states.shape, face_hidden_states.shape)
        # import pdb;pdb.set_trace()
        # hidden_states = body_hidden_states + attn.ip_scale * repeat(head_hidden_states.detach(), 'b l c -> (b v) l c', v=n_view)      
        # hidden_states = rearrange(
        #     torch.cat([rearrange(hidden_states, '(b v) l c -> b v l c'), head_hidden_states.unsqueeze(1)], dim=1),
        #     'b v l c -> (b v) l c')
        
        # face cross attention
        # ip_hidden_states = repeat(face_hidden_states.detach(), 'b l c -> (b v) l c', v=num_views-1)
        # ip_key = attn.to_k_ip(ip_hidden_states)
        # ip_value = attn.to_v_ip(ip_hidden_states)
        # ip_key = attn.head_to_batch_dim(ip_key).contiguous()
        # ip_value = attn.head_to_batch_dim(ip_value).contiguous()
        # ip_query = attn.head_to_batch_dim(body_hidden_states).contiguous()
        # ip_hidden_states = xformers.ops.memory_efficient_attention(ip_query, ip_key, ip_value, attn_bias=attention_mask)
        # ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)
        # ip_hidden_states = attn.to_out_ip[0](ip_hidden_states)
        # ip_hidden_states = attn.to_out_ip[1](ip_hidden_states)
        # import pdb;pdb.set_trace()
        
        
        def transpose(tensor):
            tensor_0, tensor_1 = torch.chunk(tensor, dim=0, chunks=2)  # bv hw c
            tensor = torch.cat([tensor_0, tensor_1], dim=1)  # bv 2hw c
            # tensor = rearrange(tensor, "(b v) l c -> b v l c", v=num_views+1)
            # body, face = tensor[:, :-1, :], tensor[:, -1:, :] # b,v,l,c;  b,1,l,c
            # face = face.repeat(1, num_views, 1, 1)  # b,v,l,c
            # tensor = torch.cat([body, face], dim=2)  # b, v, 4hw, c 
            # tensor = rearrange(tensor, "b v l c -> (b v) l c")
            return tensor 
        key  = transpose(key)
        value = transpose(value)
        query = transpose(query)
        
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states_normal, hidden_states_color = torch.chunk(hidden_states, dim=1, chunks=2) # bv, hw, c
        
        hidden_states_normal = rearrange(hidden_states_normal, "(b v) (h w) c -> b v h w c", v=num_views+1, h=height)
        face_normal = rearrange(hidden_states_normal[:, -1, :, :, :], 'b h w c -> b c h w').detach() 
        face_normal = rearrange(F.interpolate(face_normal, size=(height_st, height_st), mode='bilinear'), 'b c h w -> b h w c')
        hidden_states_normal = hidden_states_normal.clone()  # Create a copy of hidden_states_normal
        hidden_states_normal[:, 0, :height_st, height_st:height_end, :] = 0.5 * hidden_states_normal[:, 0, :height_st, height_st:height_end, :] +  0.5 * face_normal
        # hidden_states_normal[:, 0, :height_st, height_st:height_end, :] = 0.1 * hidden_states_normal[:, 0, :height_st, height_st:height_end, :] +  0.9 * face_normal
        hidden_states_normal = rearrange(hidden_states_normal, "b v h w c -> (b v) (h w) c")

        
        hidden_states_color = rearrange(hidden_states_color, "(b v) (h w) c -> b v h w c", v=num_views+1, h=height)
        face_color = rearrange(hidden_states_color[:, -1, :, :, :], 'b h w c -> b c h w').detach()
        face_color = rearrange(F.interpolate(face_color, size=(height_st, height_st), mode='bilinear'), 'b c h w -> b h w c')
        hidden_states_color = hidden_states_color.clone()  # Create a copy of hidden_states_color
        hidden_states_color[:, 0, :height_st, height_st:height_end, :] = 0.5 * hidden_states_color[:, 0, :height_st, height_st:height_end, :] +  0.5 * face_color
        # hidden_states_color[:, 0, :height_st, height_st:height_end, :] = 0.1 * hidden_states_color[:, 0, :height_st, height_st:height_end, :] + 0.9 * face_color
        hidden_states_color = rearrange(hidden_states_color, "b v h w c -> (b v) (h w) c")
        
        hidden_states = torch.cat([hidden_states_normal, hidden_states_color], dim=0)  # 2bv hw c
        
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states
    
    def __call__(
        self,
        attn: Attention,
        hidden_states,  
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        num_tasks=2,
    ):
        hidden_states = self.process(attn, hidden_states, encoder_hidden_states, attention_mask, temb, num_tasks)
        # hidden_states = rearrange(hidden_states, '(b v) l c -> b v l c')
        # body_hidden_states, head_hidden_states = rearrange(hidden_states[:, :-1, :, :], 'b v l c -> (b v) l c'), hidden_states[:, -1:, :, :]
        # import pdb;pdb.set_trace()
        # hidden_states = body_hidden_states + attn.ip_scale * head_hidden_states.detach().repeat(1, views, 1, 1)        
        # hidden_states = rearrange(
        #     torch.cat([rearrange(hidden_states, '(b v) l c -> b v l c'), head_hidden_states], dim=1),
        #     'b v l c -> (b v) l c')
        return hidden_states
    
class IPCrossAttn(Attention):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self,  
            query_dim, cross_attention_dim, heads, dim_head, dropout, bias, upcast_attention, ip_scale=1.0):
        super().__init__(query_dim, cross_attention_dim, heads, dim_head, dropout, bias, upcast_attention)

        self.ip_scale = ip_scale
        # self.num_tokens = num_tokens

        # self.to_k_ip = nn.Linear(query_dim, self.inner_dim, bias=False)
        # self.to_v_ip = nn.Linear(query_dim, self.inner_dim, bias=False)
        
        # self.to_out_ip = nn.ModuleList([])
        # self.to_out_ip.append(nn.Linear(self.inner_dim, self.inner_dim, bias=bias))
        # self.to_out_ip.append(nn.Dropout(dropout))
        # nn.init.zeros_(self.to_k_ip.weight.data)
        # nn.init.zeros_(self.to_v_ip.weight.data)
        
    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, *args, **kwargs
    ):
        processor = XFormersIPCrossAttnProcessor()
        self.set_processor(processor)

class XFormersIPCrossAttnProcessor:   

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        num_views=1
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # ip attn
        # hidden_states = rearrange(hidden_states, '(b v) l c -> b v l c', v=num_views)
        # body_hidden_states, face_hidden_states = rearrange(hidden_states[:, :-1, :, :], 'b v l c -> (b v) l c'), hidden_states[:, -1, :, :]
        # print(body_hidden_states.shape, face_hidden_states.shape)
        # import pdb;pdb.set_trace()
        # hidden_states = body_hidden_states + attn.ip_scale * repeat(head_hidden_states.detach(), 'b l c -> (b v) l c', v=n_view)      
        # hidden_states = rearrange(
        #     torch.cat([rearrange(hidden_states, '(b v) l c -> b v l c'), head_hidden_states.unsqueeze(1)], dim=1),
        #     'b v l c -> (b v) l c')
        
        # face cross attention
        # ip_hidden_states = repeat(face_hidden_states.detach(), 'b l c -> (b v) l c', v=num_views-1)
        # ip_key = attn.to_k_ip(ip_hidden_states)
        # ip_value = attn.to_v_ip(ip_hidden_states)
        # ip_key = attn.head_to_batch_dim(ip_key).contiguous()
        # ip_value = attn.head_to_batch_dim(ip_value).contiguous()
        # ip_query = attn.head_to_batch_dim(body_hidden_states).contiguous()
        # ip_hidden_states = xformers.ops.memory_efficient_attention(ip_query, ip_key, ip_value, attn_bias=attention_mask)
        # ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)
        # ip_hidden_states = attn.to_out_ip[0](ip_hidden_states)
        # ip_hidden_states = attn.to_out_ip[1](ip_hidden_states)
        # import pdb;pdb.set_trace()
        
        # body_hidden_states = body_hidden_states + attn.ip_scale * ip_hidden_states
        # hidden_states = rearrange(
        #                     torch.cat([rearrange(body_hidden_states, '(b v) l c -> b v l c', v=num_views-1), face_hidden_states.unsqueeze(1)], dim=1),
        #                     'b v l c -> (b v) l c')
        # import pdb;pdb.set_trace()
        # 
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        

        # TODO: region control  
        # region control
        # if len(region_control.prompt_image_conditioning) == 1:
        #     region_mask = region_control.prompt_image_conditioning[0].get('region_mask', None)
        #     if region_mask is not None:
        #         h, w = region_mask.shape[:2]
        #         ratio = (h * w / query.shape[1]) ** 0.5
        #         mask = F.interpolate(region_mask[None, None], scale_factor=1/ratio, mode='nearest').reshape([1, -1, 1])
        #     else:
        #         mask = torch.ones_like(ip_hidden_states)
        #     ip_hidden_states = ip_hidden_states * mask     

        return hidden_states
    

class RowwiseMVProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        num_views=1,
        cd_attention_mid=False
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        height = int(math.sqrt(sequence_length)) 
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # print('query', query.shape, 'key', key.shape, 'value', value.shape)
        #([bx4, 1024, 320]) key torch.Size([bx4, 1024, 320]) value torch.Size([bx4, 1024, 320])
        # pdb.set_trace()
        # multi-view self-attention
        def transpose(tensor):
            tensor = rearrange(tensor, "(b v) (h w) c -> b v h w c", v=num_views, h=height)
            tensor_0, tensor_1 = torch.chunk(tensor, dim=0, chunks=2)  # b v h w c
            tensor = torch.cat([tensor_0, tensor_1], dim=3)  # b v h 2w c
            tensor = rearrange(tensor, "b v h w c -> (b h) (v w) c", v=num_views, h=height)
            return tensor

        if cd_attention_mid:
            key = transpose(key)
            value = transpose(value)
            query = transpose(query)
        else:
            key = rearrange(key, "(b v) (h w) c -> (b h) (v w) c", v=num_views, h=height) 
            value = rearrange(value, "(b v) (h w) c -> (b h) (v w) c", v=num_views, h=height)
            query = rearrange(query, "(b v) (h w) c -> (b h) (v w) c", v=num_views, h=height) # torch.Size([192, 384, 320])

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if cd_attention_mid:
            hidden_states = rearrange(hidden_states, "(b h) (v w) c -> b v h w c", v=num_views, h=height)
            hidden_states_0, hidden_states_1 = torch.chunk(hidden_states, dim=3, chunks=2)  # b v h w c
            hidden_states = torch.cat([hidden_states_0, hidden_states_1], dim=0)  # 2b v h w c
            hidden_states = rearrange(hidden_states, "b v h w c -> (b v) (h w) c", v=num_views, h=height) 
        else:
            hidden_states = rearrange(hidden_states, "(b h) (v w) c -> (b v) (h w) c", v=num_views, h=height)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states
    
    
class CDAttention(Attention):
    # def __init__(self, ip_scale,
    #             query_dim, heads, dim_head, dropout, bias, cross_attention_dim, upcast_attention, processor):
    #     super().__init__(query_dim, cross_attention_dim, heads, dim_head, dropout, bias, upcast_attention, processor=processor)    
        
        # self.ip_scale = ip_scale
        
        # self.to_k_ip = nn.Linear(query_dim, self.inner_dim, bias=False)
        # self.to_v_ip = nn.Linear(query_dim, self.inner_dim, bias=False)
        # nn.init.zeros_(self.to_k_ip.weight.data)
        # nn.init.zeros_(self.to_v_ip.weight.data)
        
        
    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, *args, **kwargs
    ):
        processor = XFormersCDAttnProcessor()
        self.set_processor(processor)
        # print("using xformers attention processor")    
    
class XFormersCDAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        num_tasks=2
    ):
        
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)


        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        assert num_tasks == 2  # only support two tasks now

        def transpose(tensor):
            tensor_0, tensor_1 = torch.chunk(tensor, dim=0, chunks=2)  # bv hw c
            tensor = torch.cat([tensor_0, tensor_1], dim=1)  # bv 2hw c
            return tensor 
        key  = transpose(key)
        value = transpose(value)
        query = transpose(query)

        
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = torch.cat([hidden_states[:, 0], hidden_states[:, 1]], dim=0)  # 2bv hw c
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states
   