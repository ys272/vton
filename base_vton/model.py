import matplotlib.pyplot as plt
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
        init_dim=32,
        level_dims=(32, 48, 64),
        level_dims_cross_attn=(32, 48, 64),
        level_attentions=(False, True),
        level_repetitions = (2,3,4),
        channels=3,
        pose_dim=34,
    ):
        super().__init__()
        
        self.level_attentions = level_attentions
        self.level_dims = level_dims
        self.level_repetitions = level_repetitions
                
        # film embeddings
        individual_film_dim = init_dim
        combined_film_dim = 2 * individual_film_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(init_dim),
            nn.Linear(init_dim, individual_film_dim),
            nn.SiLU(),
            nn.Linear(individual_film_dim, individual_film_dim),
            # nn.SiLU(),
            # nn.Linear(individual_film_dim, individual_film_dim),
        )
        self.masked_person_pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, individual_film_dim),
            nn.SiLU(),
            nn.Linear(individual_film_dim, individual_film_dim),
            # nn.SiLU(),
            # nn.Linear(individual_film_dim, individual_film_dim),
        )
        # self.masked_person_aug_mlp = nn.Sequential(
        #     SinusoidalPositionEmbeddings(init_dim),
        #     nn.Linear(init_dim, individual_film_dim),
        #     nn.SiLU(),
        #     nn.Linear(individual_film_dim, individual_film_dim),
        #     # nn.SiLU(),
        #     # nn.Linear(individual_film_dim, individual_film_dim),
        # )
        
        self.combined_embedding_masked_person = nn.Sequential(
            nn.Linear(combined_film_dim, combined_film_dim),
            nn.SiLU(),
            nn.Linear(combined_film_dim, combined_film_dim),
            # nn.SiLU(),
            # nn.Linear(combined_film_dim, combined_film_dim),
        )
        
        self.init_conv = nn.Conv2d(channels, init_dim, 3, padding=1)

        self.downs = nn.ModuleList([])
        self.mid1 = None
        self.mid2 = None
        self.ups = nn.ModuleList([])
        
        # Down levels
        for level_idx in range(len(level_dims)-1):
            dim_in = init_dim if level_idx == 0 else level_dims[level_idx-1]
            dim_out = level_dims[level_idx]
            dim_out_cross_attn = level_dims_cross_attn[level_idx]
            dim_next = level_dims[level_idx+1]
            level_att = level_attentions[level_idx]
            level_reps = level_repetitions[level_idx]
            layers = []
            for rep in range(level_reps):
                layers.append(ResnetBlock(init_dim if level_idx==0 and rep==0 else dim_out, dim_out, film_emb_dim=combined_film_dim))
                if level_att:
                    layers.append(Residual(PreNorm(SelfAttention(dim_out), dim_out), dim=None))
                    layers.append(Residual(PreNorm(CrossAttention(dim_out, dim_out_cross_attn, dim_head=64), dim_out, dim_out_cross_attn, affine=False), dim=dim_out))
            layers.append(Downsample(dim_out, dim_next))
            self.downs.append(nn.ModuleList(layers))

        # Middle level
        # First half
        dim_out = level_dims[-1]
        dim_out_cross_attn = level_dims_cross_attn[-1]
        level_reps = level_repetitions[-1]
        layers = []
        for rep in range(level_reps):
            layers.append(ResnetBlock(dim_out, dim_out, film_emb_dim=combined_film_dim))
            layers.append(Residual(PreNorm(SelfAttention(dim_out), dim_out), dim=None))
            layers.append(Residual(PreNorm(CrossAttention(dim_out, dim_out_cross_attn, dim_head=64), dim_out, dim_out_cross_attn, affine=False), dim=dim_out))
        self.mid1 = nn.ModuleList(layers)
        
        # Second half
        layers = []
        for rep in range(level_reps):
            layers.append(ResnetBlock(dim_out if rep==0 else dim_out*2, dim_out, film_emb_dim=combined_film_dim))
            layers.append(Residual(PreNorm(SelfAttention(dim_out), dim_out), dim=None))
            layers.append(Residual(PreNorm(CrossAttention(dim_out, dim_out_cross_attn, dim_head=64), dim_out, dim_out_cross_attn, affine=False), dim=dim_out))
        self.mid2 = nn.ModuleList(layers)

        # Up level
        for level_idx in range(len(level_dims)-2,-1,-1):
            dim_in = level_dims[level_idx+1]
            dim_out = level_dims[level_idx]
            dim_out_cross_attn = level_dims_cross_attn[level_idx]
            level_att = level_attentions[level_idx]
            level_reps = level_repetitions[level_idx]
            layers = []
            layers.append(Upsample(dim_in, dim_in))
            for rep in range(level_reps):
                layers.append(ResnetBlock(dim_in+dim_out if rep==0 else 2*dim_out, dim_out, film_emb_dim=combined_film_dim))
                if level_att:
                    layers.append(Residual(PreNorm(SelfAttention(dim_out), dim_out), dim=None))
                    layers.append(Residual(PreNorm(CrossAttention(dim_out, dim_out_cross_attn, dim_head=64), dim_out, dim_out_cross_attn, affine=False), dim=dim_out))
            self.ups.append(nn.ModuleList(layers))

        self.final_res_block = ResnetBlock(level_dims[0]+init_dim, level_dims[0], film_emb_dim=combined_film_dim)
        self.final_res_block = nn.Conv2d(level_dims[0]+init_dim, level_dims[0], 3, padding=1)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(level_dims[0], 3, 3, padding=1)


    def forward(self, masked_aug, pose, noise_amount_masked, t, cross_attns=None):
        x = self.init_conv(masked_aug)
        r = x.clone()
        if c.REVERSE_DIFFUSION_SAMPLER == 'karras':
            time_vector = self.time_mlp((t * c.NUM_TIMESTEPS / c.KARRAS_SIGMA_MAX).int())
        else:
            time_vector = self.time_mlp(t)
        pose_vector = self.masked_person_pose_mlp(pose)
        # noise_vector = self.masked_person_aug_mlp(noise_amount_masked)
        # film_vector = self.combined_embedding_masked_person(torch.cat((time_vector, pose_vector, noise_vector), dim=1))
        film_vector = self.combined_embedding_masked_person(torch.cat((time_vector, pose_vector), dim=1))
        # film_vector = torch.cat((time_vector, pose_vector), dim=1)
        
        cross_attn_idx = 0
        
        h = []

        for level_idx in range(len(self.downs)):
            level_att = self.level_attentions[level_idx]
            if level_att:
                for layer_idx in range(0, len(self.downs[level_idx])-1, 3):
                    res_block = self.downs[level_idx][layer_idx]
                    self_attention = self.downs[level_idx][layer_idx+1]
                    cross_attention = self.downs[level_idx][layer_idx+2]
                    x = res_block(x, film_vector)
                    h.append(x)
                    x = self_attention(x)
                    x = cross_attention(x, cross_attns[cross_attn_idx])
                    cross_attn_idx += 1
            else:
                for layer_idx in range(0, len(self.downs[level_idx])-1):
                    res_block = self.downs[level_idx][layer_idx]
                    x = res_block(x, film_vector)
                    h.append(x)
            downsample = self.downs[level_idx][-1]
            x = downsample(x)
        
        h_middle = []
        for mid_layer_idx in range(0, len(self.mid1), 3):
            res_block = self.mid1[mid_layer_idx]
            self_attention = self.mid1[mid_layer_idx+1]
            cross_attention = self.mid1[mid_layer_idx+2]
            x = res_block(x, film_vector)
            x = self_attention(x)
            x = cross_attention(x, cross_attns[cross_attn_idx])
            cross_attn_idx += 1
            if mid_layer_idx != len(self.mid1) - 2:
                h_middle.append(x)
        
        for mid_layer_idx in range(0, len(self.mid2), 3):
            res_block = self.mid2[mid_layer_idx]
            self_attention = self.mid2[mid_layer_idx+1]
            cross_attention = self.mid2[mid_layer_idx+2]
            if mid_layer_idx != 0:
                x = torch.cat((x, h_middle.pop()), dim=1)
            x = res_block(x, film_vector)
            x = self_attention(x)
            x = cross_attention(x, cross_attns[cross_attn_idx])
            cross_attn_idx += 1
                
        for level_idx in range(len(self.ups)):
            level_att = self.level_attentions[len(self.level_attentions) - 1 - level_idx]
            upsample = self.ups[level_idx][0]
            x = upsample(x)
            if level_att:
                for layer_idx in range(1, len(self.ups[level_idx]), 3):
                    res_block = self.ups[level_idx][layer_idx]
                    self_attention = self.ups[level_idx][layer_idx+1]
                    cross_attention = self.ups[level_idx][layer_idx+2]
                    x = torch.cat((x, h.pop()), dim=1)
                    x = res_block(x, film_vector)
                    x = self_attention(x)
                    x = cross_attention(x, cross_attns[cross_attn_idx])
                    cross_attn_idx += 1
            else:
                for layer_idx in range(1, len(self.ups[level_idx])):
                    res_block = self.ups[level_idx][layer_idx]
                    x = torch.cat((x, h.pop()), dim=1)
                    x = res_block(x, film_vector)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x)
        x = self.final_act(x)
        
        return self.final_conv(x)


class Unet_Clothing(nn.Module):
    def __init__(
        self,
        init_dim=16,
        level_dims=(32, 48, 64),
        level_repetitions = (2,3,4),
        channels=3,
        pose_dim=34
    ):
        super().__init__()
        
        self.level_dims = level_dims
        self.level_repetitions = level_repetitions
        
        # film embeddings
        individual_film_dim = init_dim
        combined_film_dim = individual_film_dim       
        
        self.masked_person_pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, individual_film_dim),
            nn.SiLU(),
            nn.Linear(individual_film_dim, individual_film_dim),
            # nn.SiLU(),
            # nn.Linear(individual_film_dim, individual_film_dim),
        )
        # self.clothing_aug_mlp = nn.Sequential(
        #     SinusoidalPositionEmbeddings(init_dim),
        #     nn.Linear(init_dim, individual_film_dim),
        #     nn.SiLU(),
        #     nn.Linear(individual_film_dim, individual_film_dim),
            # nn.SiLU(),
            # nn.Linear(individual_film_dim, individual_film_dim),
        # )
        # self.combined_embedding_clothing = nn.Sequential(
        #     nn.Linear(combined_film_dim, combined_film_dim),
        #     nn.SiLU(),
        #     nn.Linear(combined_film_dim, combined_film_dim),
        #     nn.SiLU(),
        #     nn.Linear(combined_film_dim, combined_film_dim),
        # )
        
        self.init_conv = nn.Conv2d(channels, init_dim, 3, padding=1)

        self.downs = nn.ModuleList([])
        self.mid1 = None
        self.mid2 = None
        self.ups = nn.ModuleList([])
        
        # Down levels
        for level_idx in range(len(level_dims)-1):
            dim_in = init_dim if level_idx == 0 else level_dims[level_idx-1]
            dim_out = level_dims[level_idx]
            dim_next = level_dims[level_idx+1]
            level_reps = level_repetitions[level_idx]
            layers = []
            for rep in range(level_reps):
                layers.append(ResnetBlock(init_dim if level_idx==0 and rep==0 else dim_out, dim_out, clothing=True if level_idx==len(level_dims)-2 else False))
            layers.append(Downsample(dim_out, dim_next))
            self.downs.append(nn.ModuleList(layers))

        # Middle level
        # First half
        dim_out = level_dims[-1]
        level_reps = level_repetitions[-1]
        layers = []
        for rep in range(level_reps):
            layers.append(ResnetBlock(dim_out, dim_out, clothing=True))
        self.mid1 = nn.ModuleList(layers)
        
        # Second half
        layers = []
        for rep in range(level_reps):
            layers.append(ResnetBlock(dim_out if rep==0 else dim_out*2, dim_out))
        self.mid2 = nn.ModuleList(layers)

        # Up level
        for level_idx in [len(level_dims)-2]:
            dim_in = level_dims[level_idx+1]
            dim_out = level_dims[level_idx]
            level_reps = level_repetitions[level_idx]
            layers = []
            layers.append(Upsample(dim_in, dim_in))
            for rep in range(level_reps):
                layers.append(ResnetBlock(dim_in+dim_out if rep==0 else 2*dim_out, dim_out))
            self.ups.append(nn.ModuleList(layers))


    def forward(self, clothing_aug, pose, noise_amount_clothing):
        x = self.init_conv(clothing_aug)
        pose_vector = self.masked_person_pose_mlp(pose)
        # noise_vector = self.clothing_aug_mlp(noise_amount_clothing)
        # film_vector = self.combined_embedding_clothing(torch.cat((pose_vector, noise_vector), dim=1))
        film_vector = pose_vector

        h = []
        
        for level_idx in range(len(self.downs)):
            for layer_idx in range(0, len(self.downs[level_idx])-1):
                res_block = self.downs[level_idx][layer_idx]
                x = res_block(x, film_vector)
                if level_idx == len(self.downs) - 1:
                    h.append(x)
            downsample = self.downs[level_idx][-1]
            x = downsample(x)

        for mid_layer_idx in range(len(self.mid1)):
            res_block = self.mid1[mid_layer_idx]
            x = res_block(x, film_vector)
            h.append(x)
        
        h_idx = len(h) - 1 
                     
        for mid_layer_idx in range(len(self.mid2)):
            res_block = self.mid2[mid_layer_idx]
            if mid_layer_idx != 0:
                x = torch.cat((x, h[h_idx]), dim=1)
            h_idx -= 1
            x = res_block(x, film_vector)
            h.append(x)
                
        for level_idx in range(len(self.ups)):
            upsample = self.ups[level_idx][0]
            x = upsample(x)
            for layer_idx in range(1, len(self.ups[level_idx])):
                res_block = self.ups[level_idx][layer_idx]
                x = torch.cat((x, h[h_idx]), dim=1)
                h_idx -=1
                x = res_block(x, film_vector)
                h.append(x)

        return h

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
