import cv2
import numpy as np
import random
import os
import math
import torch
from torch import nn
import config as c
import matplotlib.pyplot as plt
from inspect import isfunction
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from functools import partial
from torch import nn, einsum
import torch.nn.functional as F
from diffusion_ddim import q_sample, extract, alphas_cumprod
from diffusion_karras import q_sample_karras, scalings_karras
from utils import preprocess_s_person_output_for_m


'''
Generic helper funcs.
'''

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


'''
Positional embedding with sin and cos.
'''
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        max_period =c.NUM_DIFFUSION_TIMESTEPS
        embeddings = math.log(max_period) / (half_dim -1)
        embeddings = torch.exp(torch.arange(half_dim, device=c.DEVICE) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# a=SinusoidalPositionEmbeddings(32)
# aa=a.forward(torch.arange(0, 1000).long())
# plt.imshow(aa.T, cmap='gray')
# plt.show(block=True)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, y=None, *args, **kwargs):
        if y is None:
            out = self.fn(x, *args, **kwargs)
        else:
            out = self.fn(x, y, *args, **kwargs)        
        
        return out + x
    

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # max pooling could be achieved as follows:
    # reduce(x, 'b c (h h1) (w w1) -> b c h w', reduction='max', h1=2, w1=2)
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )
    
    
class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        '''
        TODO: At inference time the mean and var are fixed, so this shouldn't have to be computed every time.
        Need to investigate this more, but perhaps this be achieved using buffers that store the mean and var. 
        During training those will be updated at each iteration, and during inference they will be read from, 
        without performing any calculations.
        '''
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, up64=False):
        super().__init__()
        groups = min(32, dim//4)
        self.norm = nn.GroupNorm(groups, dim)
        self.act = nn.SiLU()
        # self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.proj = nn.Conv2d(dim, dim_out, 3 if not up64 else 5, padding=1 if not up64 else 2)
   
    def forward(self, x, scale_shift=None):
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            # TODO: Experiment if removing the +1 makes a meaningful difference.
            x = x * (scale + 1) + shift        

        x = self.act(x)
        
        x = self.proj(x)
        return x
    
    
class BlockClothing(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        groups = min(32, dim//4)
        self.norm = nn.GroupNorm(groups, dim)
        self.act = nn.SiLU()
        dilation_rate = 2
        kernel_size = 7
        # 1/10th of the channels will be dilated convolutions.
        out_channels_dilated = dim_out//10
        padding = dilation_rate * (kernel_size - 1) // 2
        # self.proj_dilated = WeightStandardizedConv2d(dim, out_channels_dilated, kernel_size, padding=padding, dilation=dilation_rate)
        # self.proj_dense = WeightStandardizedConv2d(dim, dim_out - out_channels_dilated, 3, padding=1)
        self.proj_dilated = nn.Conv2d(dim, out_channels_dilated, kernel_size, padding=padding, dilation=dilation_rate)
        self.proj_dense = nn.Conv2d(dim, dim_out - out_channels_dilated, 3, padding=1)
        
   
    def forward(self, x, scale_shift=None):
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            # TODO: Experiment if removing the +1 makes a meaningful difference.
            x = x * (scale + 1) + shift        

        x = self.act(x)
        
        x_dense = self.proj_dense(x)
        x_dilated = self.proj_dilated(x)
        x = torch.cat((x_dense,x_dilated), dim=1)
        
        return x


'''
Resnet block.

The DDPM authors employed a Wide ResNet block (Zagoruyko et al., 2016), but Phil Wang has replaced the 
standard convolutional layer by a "weight standardized" version, which works better in combination with 
group normalization (see (Kolesnikov et al., 2019) for details).
'''

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, film_emb_dim=None, clothing=False, up64=False):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(film_emb_dim, dim * 2))
            if film_emb_dim is not None
            else None
        )
        if not clothing:
            self.block1 = Block(dim, dim_out, up64=up64)
        else:
            self.block1 = BlockClothing(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        # self.block2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        # self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.res_conv = nn.Conv2d(dim, dim_out, 1)

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


'''
Attention
'''

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        # linear projection, creating `dim_head` values for each head; for queries, keys and values.
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        # project attention values back to original dimension so it can be added to original values.
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale
        # since i==j==#elements, the resulting "b h i j" matrix dimensions are (batch, heads, # elements, # elements),
        # representing the elementwise similarity (pre softmax)
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        # subtract the max for numerical stability when computing softmax (the sum of exponentiated 0s and negative
        # values should be a relatively small number, while the original values may have been enormous)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        # linear projection, creating `dim_head` values for each head; for queries
        self.to_q = nn.Conv2d(q_dim, hidden_dim, 1, bias=False)
        # linear projection, creating `dim_head` values for each head; for keys and values.
        self.to_kv = nn.Conv2d(kv_dim, hidden_dim * 2, 1, bias=False)
        # self.to_qv = nn.Conv2d(q_dim, hidden_dim * 2, 1, bias=False)
        # # linear projection, creating `dim_head` values for each head; for keys and values.
        # self.to_k = nn.Conv2d(kv_dim, hidden_dim, 1, bias=False)
        # project attention values back to original dimension so it can be added to original values.
        self.to_out = nn.Conv2d(hidden_dim, q_dim, 1, bias=False)

    def forward(self, x_q, x_kv):
        b, c, h, w = x_q.shape
        
        q = self.to_q(x_q)
        q = rearrange(q, "b (h c) x y -> b h c (x y)", h=self.heads)
        q = q * self.scale
        
        kv = self.to_kv(x_kv).chunk(2, dim=1)
        k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), kv
        )
        
        # k = self.to_k(x_kv)
        # k = rearrange(k, "b (h c) x y -> b h c (x y)", h=self.heads)
        # k = k * self.scale
        
        # qv = self.to_qv(x_q).chunk(2, dim=1)
        # q, v = map(
        #     lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qv
        # )
        
        # since i==j==#elements, the resulting "b h i j" matrix dimensions are (batch, heads, # elements, # elements),
        # representing the elementwise similarity (pre softmax)
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        # subtract the max for numerical stability when computing softmax (the sum of exponentiated 0s and negative
        # values should be a relatively small number, while the original values may have been enormous)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


'''
A wrapper class for performing *layer* normalization. 
TODO: Note that there's been a debate about whether to apply normalization
before or after attention in Transformers.
'''
class PreNorm(nn.Module):
    def __init__(self, fn, dim_x, dim_y=None, affine=True):
        super().__init__()
        self.fn = fn
        # Since group size is 1, this is equivalent to layer normalization. 
        # Layer normalization is essentially a normalization of the entire image (across all channels),
        # unlike group normalization which is only across a subset of channels, where each subset contains 
        # a number of channels equal to num_channels / num_groups. In this function's case num_groups is 1, 
        # which as stated, makes the group normalization equivalent to layer normalization.
        self.norm_x = nn.GroupNorm(1, dim_x, affine=affine)
        if dim_y is not None:
            self.norm_y = nn.GroupNorm(1, dim_y, affine=affine)

    def forward(self, x, y=None):
        x = self.norm_x(x)
        if y is None:
            return self.fn(x)
        else:
            y = self.norm_y(y)
            return self.fn(x, y)


class EMA:
    def __init__(self, beta, batch_num_when_ema_should_start, was_i_initialized=False):
        super().__init__()
        self.beta = beta
        self.was_i_initialized = was_i_initialized
        self.batch_num_when_ema_should_start = batch_num_when_ema_should_start

    def update_model_average(self, ema_model_main, model_main, ema_model_aux, model_aux):
        for current_params, ma_params in zip(model_main.parameters(), ema_model_main.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
        for current_params, ma_params in zip(model_aux.parameters(), ema_model_aux.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model_main, model_main, ema_model_aux, model_aux, batch_num):
        if batch_num >= self.batch_num_when_ema_should_start:
            if not self.was_i_initialized:
                self.reset_parameters(ema_model_main, model_main, ema_model_aux, model_aux)
                self.was_i_initialized = True
            else:
                self.update_model_average(ema_model_main, model_main, ema_model_aux, model_aux)

    def reset_parameters(self, ema_model_main, model_main, ema_model_aux, model_aux): 
        ema_model_main.load_state_dict(model_main.state_dict())
        ema_model_aux.load_state_dict(model_aux.state_dict())
        
        
class TrainerHelper:
    def __init__(self, human_readable_timestamp, min_loss=float('inf'), min_loss_batch_num=0, last_save_batch_num=0):
        self.min_loss = min_loss
        self.min_loss_batch_num = min_loss_batch_num
        self.human_readable_timestamp = human_readable_timestamp
        self.last_learning_rate_reduction = 0
        self.last_accumulation_rate_increase = 0
        self.last_save_batch_num = last_save_batch_num
        
    def save(self, min_loss, model_main, model_aux, ema_model_main, ema_model_aux, was_ema_initialized, optimizer, scaler, batch_num, accumulation_rate, epoch, validation_dataset_start_idx, save_from_this_batch_num=0, suffix=''):
        save_path = os.path.join(c.MODEL_OUTPUT_PARAMS_DIR, self.human_readable_timestamp + suffix)
        torch.save({
            'batch_num': batch_num,
            'epoch': epoch,
            'model_main_state_dict': model_main.state_dict() if model_main else None,
            'model_aux_state_dict': model_aux.state_dict()  if model_aux else None,
            'model_ema_main_state_dict': ema_model_main.state_dict() if ema_model_main else None,
            'model_ema_aux_state_dict': ema_model_aux.state_dict() if ema_model_aux else None,
            'was_ema_initialized': was_ema_initialized,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'loss': min_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'accumulation_rate': accumulation_rate,
            'last_accumulation_rate_increase': self.last_accumulation_rate_increase,
            'last_learning_rate_reduction': self.last_learning_rate_reduction,
            'last_save_batch_num': self.last_save_batch_num,
            'validation_dataset_start_idx': validation_dataset_start_idx,
        }, save_path)
                
    def update_loss_possibly_save_model(self, loss, model_main, model_aux, ema_model_main, ema_model_aux, was_ema_initialized, optimizer, scaler, batch_num, accumulation_rate, epoch, validation_dataset_start_idx, save_from_this_batch_num=0):
        if loss < self.min_loss:
            self.min_loss = loss
            self.min_loss_batch_num = batch_num
            if batch_num >= save_from_this_batch_num:
                save_suffix = f'_MIN_loss.pth'
                self.last_save_batch_num = batch_num
                self.save(self.min_loss, model_main, model_aux, ema_model_main, ema_model_aux, was_ema_initialized, optimizer, scaler, batch_num, accumulation_rate, epoch, validation_dataset_start_idx, save_from_this_batch_num=save_from_this_batch_num, suffix=save_suffix)
        elif self.last_save_batch_num != 0 and (batch_num - self.last_save_batch_num) > c.FREQUENCY_SAVE_MODEL_WITHOUT_LOSS_DECREASE:
            save_suffix = f'_{batch_num}_normal_loss_{loss:.3f}.pth'
            self.last_save_batch_num = batch_num
            self.save(self.min_loss, model_main, model_aux, ema_model_main, ema_model_aux, was_ema_initialized, optimizer, scaler, batch_num, accumulation_rate, epoch, validation_dataset_start_idx, save_from_this_batch_num=save_from_this_batch_num, suffix=save_suffix)
        return batch_num - self.min_loss_batch_num
    
    
    def update_last_learning_rate_reduction(self, batch_num):
        self.last_learning_rate_reduction = batch_num        
        
    def update_last_accumulation_rate_increase(self, batch_num):
        self.last_accumulation_rate_increase = batch_num      
    
    def num_batches_since_last_learning_rate_reduction(self, batch_num):
        return batch_num - self.last_learning_rate_reduction
    
    def num_batches_since_last_accumulation_rate_increase(self, batch_num):
        return batch_num - self.last_accumulation_rate_increase


def p_losses(model_main, model_aux, clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, noise_amount_clothing, noise_amount_masked, t, noise=None, loss_type="l1", apply_cfg=False):
    if noise is None:
        noise = torch.randn_like(masked_aug) * c.NOISE_SCALING_FACTOR

    if c.REVERSE_DIFFUSION_SAMPLER == 'karras':
        x_noisy, noise = q_sample_karras(person, t, noise=noise)
    else:
        x_noisy = q_sample(person, t=t, noise=noise)
    
    if c.USE_CLASSIFIER_FREE_GUIDANCE and apply_cfg:
        level_dims_aux = c.MODELS_PARAMS[c.IMAGE_SIZE][1]
        # cross_attns = [len(model_aux.mid2) * [torch.zeros((clothing_aug.shape[0], level_dims_aux[-1], 16, 11), device=c.DEVICE)], (len(model_aux.ups[0])-1) * [torch.zeros((clothing_aug.shape[0], level_dims_aux[-2], 32, 22), device=c.DEVICE)]]
        # cross_attns = [*cross_attns[0],*cross_attns[1]]
        cross_attns = [torch.zeros((clothing_aug.shape[0], level_dims_aux[-1], 16, 11), device=c.DEVICE), torch.zeros((clothing_aug.shape[0], level_dims_aux[-2], 32, 22), device=c.DEVICE)]
        mask_coords = torch.zeros_like(mask_coords)
        masked_aug = torch.zeros_like(masked_aug)
        pose_vector = torch.zeros_like(pose_vector)
    else:
        cross_attns = model_aux(clothing_aug, pose_vector, noise_amount_clothing, t)
    
    x_noisy_and_masked_aug = torch.cat((x_noisy, masked_aug, pose_matrix, mask_coords.to(clothing_aug.dtype).unsqueeze(1)), dim=1)
    predicted_noise = model_main(x_noisy_and_masked_aug, pose_vector, noise_amount_masked, t, cross_attns=cross_attns)
    alphas_squared = extract(alphas_cumprod, t, t.shape) ** 2
    snr = alphas_squared / (1 - alphas_squared)
    # Recommended gamma values ranged from 5 to 20. I chose 10 here.
    min_snr_gamma_weight = torch.minimum(torch.ones_like(snr), 10 / snr)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise, reduction='none')
        if c.REVERSE_DIFFUSION_SAMPLER == 'ddim' and c.USE_MIN_SNR_GAMMA_WEIGHTING:
            loss_per_sample = torch.mean(loss, dim=(1,2,3))
            weighted_loss = loss_per_sample * min_snr_gamma_weight
            loss = weighted_loss.mean()
        else:
            loss = loss.mean()
        # mask_coords = mask_coords.unsqueeze(1).expand(-1, 3, -1, -1)
        # loss[~mask_coords] = 0
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise, reduction='none')
        # mask_coords = mask_coords.unsqueeze(1).expand(-1, 3, -1, -1)
        # loss[~mask_coords] = 0
        loss = loss.mean()
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()    

    return loss


def p_losses_autoencoder(model_aux, clothing_aug, loss_type="l1"):
    reconstructed_image = model_aux(clothing_aug)
    if loss_type == 'l1':
        loss = F.l1_loss(reconstructed_image, clothing_aug, reduction='mean')
    elif loss_type == 'l2':
        loss = F.mse_loss(reconstructed_image, clothing_aug, reduction='mean')
    return loss