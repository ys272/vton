import math
from inspect import isfunction
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchviz import make_dot
from nn_utils import *


# def nonlinearity(x):
#     return F.silu(x)


# def normalize(x, name):
#     return nn.GroupNorm(num_groups=32, num_channels=x.size(1))


# def upsample(x, name, with_conv):
#     B, C, H, W = x.size()
#     x = F.interpolate(x, scale_factor=2, mode='nearest')
#     assert x.size() == (B, C, H * 2, W * 2)
#     if with_conv:
#         x = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)(x)
#         assert x.size() == (B, C, H * 2, W * 2)
#     return x


# def downsample(x, name, with_conv):
#     B, C, H, W = x.size()
#     if with_conv:
#         x = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1)(x)
#     else:
#         x = F.avg_pool2d(x, kernel_size=2, stride=2)
#     assert x.size() == (B, C, H // 2, W // 2)
#     return x


# def resnet_block(x, temb, name, out_ch=None, conv_shortcut=False, dropout=0.):
#     B, C, H, W = x.size()
#     if out_ch is None:
#         out_ch = C

#     h = x

#     h = nonlinearity(normalize(h, name='norm1'))
#     h = nn.Conv2d(C, out_ch, kernel_size=3, stride=1, padding=1)(h)

#     # add in timestep embedding
#     temb_proj = nn.Linear(temb.size(1), out_ch)
#     temb_proj.weight.data.fill_(0.)
#     temb_proj.bias.data.fill_(0.)
#     h += temb_proj(nonlinearity(temb)).view(B, out_ch, 1, 1)

#     h = nonlinearity(normalize(h, name='norm2'))
#     h = F.dropout(h, p=dropout)
#     h = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)(h)

#     if C != out_ch:
#         if conv_shortcut:
#             x = nn.Conv2d(C, out_ch, kernel_size=1, stride=1)(x)
#         else:
#             x = nn.Conv2d(C, out_ch, kernel_size=1, stride=1)(x)

#     assert x.size() == h.size()
#     print('{}: x={} temb={}'.format(name, x.size(), temb.size()))
#     return x + h


# def attn_block(x, name, temb):
#     B, C, H, W = x.size()

#     h = normalize(x, name='norm')
#     q = nn.Conv2d(C, C, kernel_size=1)(h)
#     k = nn.Conv2d(C, C, kernel_size=1)(h)
#     v = nn.Conv2d(C, C, kernel_size=1)(h)

#     w = torch.einsum('bhwc,bHWc->bhwHW', q, k) * (C ** (-0.5))
#     w = w.view(B, H, W, H * W)
#     w = F.softmax(w, dim=-1)
#     w = w.view(B, H, W, H, W)

#     h = torch.einsum('bhwHW,bHWc->bhwc', w, v)
#     h = nn.Conv2d(C, C, kernel_size=1)(h)

#     assert h.size() == x.size()
#     print(name, x.size())
#     return x + h


# class Model(nn.Module):
#     def __init__(self, t, name, num_classes, ch, out_ch, ch_mult, num_res_blocks,
#                  attn_resolutions, dropout=0., resamp_with_conv=True):
#         super(Model, self).__init__()
#         B, S, _, _ = t.size()
#         assert t.dtype in [torch.int32, torch.int64]
#         num_resolutions = len(ch_mult)
#         assert num_classes == 1, 'not supported'

#         self.name = name

#         # Timestep embedding
#         self.temb = nn.Sequential(
#             nn.Linear(t.size(1), ch * 4),
#             nn.SiLU(),
#             nn.Linear(ch * 4, ch * 4)
#         )

#         # Downsampling
#         self.hs = [nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)]
#         for i_level in range(num_resolutions):
#             # Residual blocks for this resolution
#             for i_block in range(num_res_blocks):
#                 h = resnet_block(self.hs[-1], self.temb, name='block_{}'.format(i_block),
#                                  out_ch=ch * ch_mult[i_level], dropout=dropout)
#                 if h.size(2) in attn_resolutions:
#                     h = attn_block(h, name='attn_{}'.format(i_block), temb=self.temb)
#                 self.hs.append(h)
#             # Downsample
#             if i_level != num_resolutions - 1:
#                 self.hs.append(downsample(self.hs[-1], name='downsample', with_conv=resamp_with_conv))

#         # Middle
#         self.h = self.hs[-1]
#         self.h = resnet_block(self.h, self.temb, name='block_1', dropout=dropout)
#         self.h = attn_block(self.h, name='attn_1', temb=self.temb)
#         self.h = resnet_block(self.h, self.temb, name='block_2', dropout=dropout)

#         # Upsampling
#         for i_level in reversed(range(num_resolutions)):
#             # Residual blocks for this resolution
#             for i_block in range(num_res_blocks + 1):
#                 h = resnet_block(torch.cat([self.h, self.hs.pop()], dim=1), name='block_{}'.format(i_block),
#                                  temb=self.temb, out_ch=ch * ch_mult[i_level], dropout=dropout)
#                 if h.size(2) in attn_resolutions:
#                     h = attn_block(h, name='attn_{}'.format(i_block), temb=self.temb)
#             # Upsample
#             if i_level != 0:
#                 self.h = upsample(self.h, name='upsample', with_conv=resamp_with_conv)

#         # End
#         self.h = nonlinearity(normalize(self.h, name='norm_out'))
#         self.h = nn.Conv2d(out_ch, out_ch, kernel_size=1)(self.h)

#     def forward(self, x):
#         return self.h

# x = torch.randn(1, 3, 64, 64)
# t = torch.tensor([[1]], dtype=torch.int32)

# model = Model(t, name='model', num_classes=1, ch=64, out_ch=3, ch_mult=(1, 2, 4, 8),
#               num_res_blocks=2, attn_resolutions=[16], dropout=0.1, resamp_with_conv=True)
# output = model(x)
# print(output.size())


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            # TODO: Consider modifying this. Essentially there is an additional block before the 
            # middle of the U, not doing any downsampling.
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# image_size = 28
# num_channels = 1
# x = torch.randn(c.BATCH_SIZE, num_channels, image_size, image_size, device=c.DEVICE)
# t = torch.randint(0, c.NUM_TIMESTEPS, (c.BATCH_SIZE,), device=c.DEVICE).long()
# num_dims_first_layer = 16
# model = Unet(num_dims_first_layer, channels=num_channels, dim_mults=(1, 2, 4))
# model.to(c.DEVICE)
# output = model(x,t)
# print(output.size())
# make_dot(model(x,t), params=dict(model.named_parameters())).render("/home/yoni/Desktop/fash_model", format="png")