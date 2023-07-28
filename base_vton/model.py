import math
from inspect import isfunction
from functools import partial
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchviz import make_dot
from nn_utils import *

    
class Unet_Person_Masked(nn.Module):
    def __init__(
        self,
        init_dim=16,
        level_dims=(32, 48, 64),
        level_attentions=(False, True),
        level_repetitions = (2,3,4),
        channels=3,
    ):
        super().__init__()
        
        self.level_attentions = level_attentions
        self.level_dims = level_dims
        self.level_repetitions = level_repetitions
                
        # time embeddings
        time_dim = init_dim * 4
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(init_dim),
            nn.Linear(init_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.masked_person_pose_mlp = None
        self.masked_person_aug_mlp = None
        self.combined_embedding_masked_person = None
        
        self.masked_person_pose_mlp = None
        self.clothing_aug_mlp = None
        self.combined_embedding_clothing = None
        
        self.init_conv = nn.Conv2d(channels, init_dim, 3, padding=1)

        self.downs = nn.ModuleList([])
        self.mid = None
        self.ups = nn.ModuleList([])
        
        # Down levels
        for level_idx in range(len(level_dims)-1):
            dim_in = init_dim if level_idx == 0 else level_dims[level_idx-1]
            dim_out = level_dims[level_idx]
            level_att = level_attentions[level_idx]
            level_reps = level_repetitions[level_idx]
            layers = []
            for rep in range(level_reps):
                layers.append(ResnetBlock(dim_in if rep==0 else dim_out, dim_out, time_emb_dim=time_dim))
                if level_att:
                    layers.append(Residual(PreNorm(dim_out, Attention(dim_out))))
            layers.append(Downsample(dim_out, dim_out))
            self.downs.append(nn.ModuleList(layers))

        # Middle level
        dim_in = level_dims[-2]
        dim_out = level_dims[-1]
        level_reps = level_repetitions[-1]
        layers = []
        for rep in range(level_reps):
            layers.append(ResnetBlock(dim_in if rep==0 else dim_out, dim_out, time_emb_dim=time_dim))
            layers.append(Residual(PreNorm(dim_out, Attention(dim_out))))
        self.mid = nn.ModuleList(layers)

        # Up level
        for level_idx in range(len(level_dims)-2,-1,-1):
            dim_in = level_dims[level_idx+1]
            dim_out = level_dims[level_idx]
            level_att = level_attentions[level_idx]
            level_reps = level_repetitions[level_idx]
            layers = []
            layers.append(Upsample(dim_in, dim_in))
            for rep in range(level_reps):
                layers.append(ResnetBlock(dim_in+dim_out if rep==0 else 2*dim_out, dim_out, time_emb_dim=time_dim))
                if level_att:
                    layers.append(Residual(PreNorm(dim_out, Attention(dim_out))))
            self.ups.append(nn.ModuleList(layers))

        self.out_dim = 3
        # self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(level_dims[0], self.out_dim, 3, padding=1)

    def forward(self, masked_aug, pose, noise_amount_masked, t):

        x = self.init_conv(masked_aug)
        # r = x.clone()
        t = self.time_mlp(t)

        h = []

        for level_idx in range(len(self.downs)):
            level_att = self.level_attentions[level_idx]
            for layer_idx in range(0, len(self.downs[level_idx])-1, 2):
                res_block = self.downs[level_idx][layer_idx]
                x = res_block(x, t)
                h.append(x)
                if level_att:
                    attention = self.downs[level_idx][layer_idx+1]
                    x = attention(x)
                else:
                    res_block = self.downs[level_idx][layer_idx+1]
                    x = res_block(x, t)
                    h.append(x)
            downsample = self.downs[level_idx][-1]
            x = downsample(x)

        for idx in range(0, len(self.mid), 2):
            res_block = self.mid[idx]
            attention = self.mid[idx+1]
            x = res_block(x, t)
            x = attention(x)

        for level_idx in range(len(self.ups)):
            level_att = self.level_attentions[len(self.level_attentions) - 1 - level_idx]
            upsample = self.ups[level_idx][0]
            x = upsample(x)
            for layer_idx in range(1, len(self.ups[level_idx]), 2):
                res_block = self.ups[level_idx][layer_idx]
                x = torch.cat((x, h.pop()), dim=1)
                x = res_block(x, t)
                if level_att:
                    attention = self.ups[level_idx][layer_idx+1]
                    x = attention(x)
                else:
                    res_block = self.ups[level_idx][layer_idx+1]
                    x = torch.cat((x, h.pop()), dim=1)
                    x = res_block(x, t)

        # x = torch.cat((x, r), dim=1)
        # x = self.final_res_block(x, t)
        
        return self.final_conv(x)


# image_size = 28
# num_channels = 1
# x = torch.randn(c.BATCH_SIZE, num_channels, image_size, image_size, device=c.DEVICE) * c.NOISE_SCALING_FACTOR
# t = torch.randint(0, c.NUM_TIMESTEPS, (c.BATCH_SIZE,), device=c.DEVICE).long()
# num_dims_first_layer = 16
# model = Unet(num_dims_first_layer, channels=num_channels, dim_mults=(1, 2, 4))
# model.to(c.DEVICE)
# output = model(x,t)
# print(output.size())
# params = dict(model.named_parameters())
# make_dot(model(x,t), params=params).render("/home/yoni/Desktop/fash_model", format="png")
