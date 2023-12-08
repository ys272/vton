from torch.nn.modules.padding import ZeroPad2d
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
from nn_utils import Residual,PreNorm,SelfAttention,Upsample


# Strided downsampling.

class ClothingAEBlockDense(nn.Module):
    def __init__(self, dim, dim_out, final=False):
        super().__init__()
        groups = min(32, dim//4)
        self.final = final
        self.norm = nn.GroupNorm(groups, dim)
        self.act = nn.SiLU()
        if not final:
          # TODO: This isn't really a projection, I just copied this line and didn't change the name.
          self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        else: #padding is handled by the caller, using `ZeroPad2d`
          self.proj = nn.Conv2d(dim, dim_out, 3, padding=0, stride=2)
   
    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        
        return self.proj(x)
    

class ClothingAEBlock(nn.Module):
    def __init__(self, dim, dim_out, direction='down', level_idx=0):
        super().__init__()
        # If there are residual connections (in the two lowest layers of the network (16 and 32)),
        # the input dimension will be larger (as the residual output is concatenated with the original layer input).
        if direction == 'up' and level_idx >= 2:
            self.block1 = ClothingAEBlockDense(dim+dim_out, dim)
        else:
            self.block1 = ClothingAEBlockDense(dim, dim)
        self.block2 = ClothingAEBlockDense(dim, dim)
        if direction == 'down':
          self.block3 = ClothingAEBlockDense(dim, dim_out, final=True)
        elif direction == 'middle' or direction == 'up':
          self.block3 = ClothingAEBlockDense(dim, dim_out)
        if direction == 'up':
            # the upsampled input has different dimensions than the output (of block 3) that will
            # be added to it, so here we modify the input to have the same dimension.
            if level_idx >= 2:
                self.res_conv = nn.Conv2d(dim+dim_out, dim_out, 1)
            else:
                self.res_conv = nn.Conv2d(dim, dim_out, 1)
        self.direction = direction
        self.level_idx = level_idx

    def forward(self, x):
        h = self.block1(x)
        if self.direction != 'up':
            # For bottom, we add the two here rather than after block3, since block3 will decrease 
            # the spatial dimensions.
            # TODO: For middle, try adding the two after block 3.
            h = self.block2(h) + x
        if self.direction == 'down':
            # A strided downsampling convolution from e.g 16 to 8, requires padding on *one* of the sides (doesn't matter which).
            if self.level_idx % 2 == 0:
                padder = torch.nn.ZeroPad2d((0,1,0,1)) # left, right, top, bottom
            else:
                padder = torch.nn.ZeroPad2d((1,0,1,0))
            h = padder(h)
        h = self.block3(h)
        if self.direction == 'up':
            h = h + self.res_conv(x)
        return h #+ self.res_conv(x)
      

# Note that the encoder may not have a sense of what is clothing and what is background, although
# it's possible that since backgrounds are always white and not in the center, it'll learn a simple
# representation by them that'll be recognized by the downstream diffusion unet.
class Clothing_Autoencoder(nn.Module):
    def __init__(
        self,
        init_dim=16,
        level_dims=(32, 48, 64),
        # level_repetitions = (2,2,2),
        channels=3,
        training_mode=True
    ):
        super().__init__()
        
        self.level_dims = level_dims
        self.training_mode = training_mode
        self.init_conv = nn.Conv2d(channels, level_dims[0], 3, padding=1)

        self.downs = nn.ModuleList([])
        self.mid = None
        self.ups = nn.ModuleList([])
        
        # Down levels
        for level_idx in range(len(level_dims)-1):
            dim_in= level_dims[level_idx]
            dim_out = level_dims[level_idx+1]
            self.downs.append(ClothingAEBlock(dim_in, dim_out, direction='down',level_idx=level_idx))

        # Middle level
        dim_in = level_dims[-1]
        self.mid = ClothingAEBlock(dim_in, dim_in, direction='middle')

        # Up level
        for level_idx in range(len(level_dims)-2,-1,-1):
            dim_in = level_dims[level_idx+1]
            dim_out = level_dims[level_idx]
            layers = []
            layers.append(Upsample(dim_in, dim_in))
            layers.append(ClothingAEBlock(dim_in, dim_out, direction='up', level_idx=level_idx))
            self.ups.append(nn.ModuleList(layers))
            
        self.final_conv_block = ClothingAEBlock(dim_out, 3, direction='middle')
        self.final_conv = nn.Conv2d(3, 3, 3, padding=1)


    def forward(self, clothing_aug):
        x = self.init_conv(clothing_aug)
        outputs = []
        h = []
        for level_idx in range(len(self.downs)):
            res_block = self.downs[level_idx]
            x = res_block(x)
            # We are appending the data right after it was downsampled. Before any additional processing. 
            # Alternatively, consider appending after the additional processing (right before subsequent downsampling).
            if level_idx > 0 and len(h) < 2:
                h.append(x)

        res_block = self.mid
        x = res_block(x)
        if not self.training_mode:
            outputs.append(x)
        
        h_idx = len(h) - 1
        for level_idx in range(len(self.ups)):
            upsample = self.ups[level_idx][0]
            res_block = self.ups[level_idx][1]
            x = upsample(x)
            if level_idx < 2:
                x = torch.cat((x, h[h_idx]), dim=1)
                h_idx -= 1
            x = res_block(x)
            if level_idx < 2:
                outputs.append(x)
            
        x = self.final_conv_block(x)
        x = self.final_conv(x)

        if self.training_mode:
            return x
        else:
            return outputs


class Clothing_Classifier(nn.Module):
    def __init__(
        self,
        init_dim=16,
        level_dims=(32, 48, 64),
        # level_repetitions = (2,2,2),
        channels=3,
    ):
        super().__init__()
        
        self.level_dims = level_dims
        self.init_conv = nn.Conv2d(channels, level_dims[0], 3, padding=1)

        self.downs = nn.ModuleList([])
        self.mid = None
        self.ups = nn.ModuleList([])
        
        # Down levels
        for level_idx in range(len(level_dims)-1):
            dim_in= level_dims[level_idx]
            dim_out = level_dims[level_idx+1]
            if level_idx == len(level_dims)-2:
                down_level_32 = nn.ModuleList([])
                down_level_32.append(ClothingAEBlock(dim_in, dim_out, direction='down',level_idx=level_idx))
                down_level_32.append(Residual(PreNorm(SelfAttention(dim_out), dim_out)))
                down_level_32.append(Residual(PreNorm(SelfAttention(dim_out), dim_out)))
                down_level_32.append(Residual(PreNorm(SelfAttention(dim_out), dim_out)))
                self.downs.append(down_level_32)
            else:
                self.downs.append(ClothingAEBlock(dim_in, dim_out, direction='down',level_idx=level_idx))
            

        # Middle level
        dim_in = level_dims[-1]
        self.mid = ClothingAEBlock(dim_in, dim_in, direction='middle')

        # Up level
        for level_idx in [len(level_dims)-2, len(level_dims)-3]:
            dim_in = level_dims[level_idx+1]
            dim_out = level_dims[level_idx]
            layers = []
            layers.append(Upsample(dim_in, dim_in))
            layers.append(ClothingAEBlock(dim_in, dim_out, direction='up', level_idx=level_idx))
            self.ups.append(nn.ModuleList(layers))
            
        self.final_conv_block = ClothingAEBlock(dim_out, 3, direction='middle')
        self.final_conv = nn.Conv2d(3, 3, 3, padding=1)


    def forward(self, clothing_aug):
        x = self.init_conv(clothing_aug)
        outputs = []
        h = []
        for level_idx in range(len(self.downs)):
            if level_idx != len(self.downs)-1:
                res_block = self.downs[level_idx]
                x = res_block(x)
                # We are appending the data right after it was downsampled. Before any additional processing. 
                # Alternatively, consider appending after the additional processing (right before subsequent downsampling).
                if level_idx > 0 and len(h) < 2:
                    h.append(x)
            else: 
                # level 32
                res_block = self.downs[level_idx][0]
                self_att_1 = self.downs[level_idx][1]
                self_att_2 = self.downs[level_idx][2]
                self_att_3 = self.downs[level_idx][3]
                x = res_block(x)
                x = self_att_1(x)
                x = self_att_2(x)
                x = self_att_3(x)

        res_block = self.mid
        x = res_block(x)
        outputs.append(x)
        
        h_idx = len(h) - 1
        for level_idx in range(len(self.ups)):
            upsample = self.ups[level_idx][0]
            res_block = self.ups[level_idx][1]
            x = upsample(x)
            x = torch.cat((x, h[h_idx]), dim=1)
            h_idx -= 1
            x = res_block(x)
            outputs.append(x)
            
        # x = self.final_conv_block(x)
        # x = self.final_conv(x)
        
        return outputs 


# Pooling downsampling.

# class ClothingAEBlockDense(nn.Module):
#     def __init__(self, dim, dim_out, final=False):
#         super().__init__()
#         groups = min(32, dim//4)
#         self.final = final
#         self.norm = nn.GroupNorm(groups, dim)
#         self.act = nn.SiLU()
#         self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
   
#     def forward(self, x):
#         x = self.norm(x)
#         x = self.act(x)
#         return self.proj(x)
    

# class ClothingAEBlock(nn.Module):
#     def __init__(self, dim, dim_out, direction='down', level_idx=0):
#         super().__init__()
#         # If there are residual connections (in the two lowest layers of the network (16 and 32)),
#         # the input dimension will be larger (as the residual output is concatenated with the original layer input).
#         if direction == 'up' and level_idx >= 2:
#             self.block1 = ClothingAEBlockDense(dim+dim_out, dim)
#         else:
#             self.block1 = ClothingAEBlockDense(dim, dim)
#         self.block2 = ClothingAEBlockDense(dim, dim)
#         if direction == 'down':
#           self.block3 = ClothingAEBlockDense(dim, dim_out, final=True)
#         elif direction == 'middle' or direction == 'up':
#           self.block3 = ClothingAEBlockDense(dim, dim_out)
#         if direction == 'up':
#             if level_idx >= 2:
#                 self.res_conv = nn.Conv2d(dim+dim_out, dim_out, 1)
#             else:
#                 self.res_conv = nn.Conv2d(dim, dim_out, 1)
#         self.direction = direction
#         self.level_idx = level_idx

#     def forward(self, x):
#         h = self.block1(x)
#         if self.direction != 'up':
#             h = self.block2(h) + x
#         h = self.block3(h)
#         if self.direction == 'up':
#             h = h + self.res_conv(x)
#         return h #+ self.res_conv(x)
      

# # Note that the encoder may not have a sense of what is clothing and what is background, although
# # it's possible that since backgrounds are always white and not in the center, it'll learn a simple
# # representation by them that'll be recognized by the downstream diffusion unet.
# class Clothing_Autoencoder(nn.Module):
#     def __init__(
#         self,
#         init_dim=16,
#         level_dims=(32, 48, 64),
#         # level_repetitions = (2,2,2),
#         channels=3
#     ):
#         super().__init__()
        
#         self.level_dims = level_dims
        
#         self.init_conv = nn.Conv2d(channels, level_dims[0], 3, padding=1)

#         self.downs = nn.ModuleList([])
#         self.mid = None
#         self.ups = nn.ModuleList([])
        
#         # Down levels
#         for level_idx in range(len(level_dims)-1):
#             dim_in= level_dims[level_idx]
#             dim_out = level_dims[level_idx+1]
#             layers = []
#             layers.append(ClothingAEBlock(dim_in, dim_in, direction='down',level_idx=level_idx))
#             layers.append(Downsample(dim_in, dim_out))
#             self.downs.append(nn.ModuleList(layers))

#         # Middle level
#         dim_in = level_dims[-1]
#         self.mid = ClothingAEBlock(dim_in, dim_in, direction='middle')

#         # Up level
#         for level_idx in range(len(level_dims)-2,-1,-1):
#             dim_in = level_dims[level_idx+1]
#             dim_out = level_dims[level_idx]
#             layers = []
#             layers.append(Upsample(dim_in, dim_in))
#             layers.append(ClothingAEBlock(dim_in, dim_out, direction='up', level_idx=level_idx))
#             self.ups.append(nn.ModuleList(layers))
            
#         self.final_conv_block = ClothingAEBlock(dim_out, 3, direction='middle')
#         self.final_conv = nn.Conv2d(3, 3, 3, padding=1)


#     def forward(self, clothing_aug):
#         x = self.init_conv(clothing_aug)
        
#         h = []
        
#         for level_idx in range(len(self.downs)):
#             res_block = self.downs[level_idx][0]
#             downsample = self.downs[level_idx][1]
#             x = res_block(x)
#             x = downsample(x)
#             # We are appending the data right after it was downsampled. Before any additional processing. 
#             # Alternatively, consider appending after the additional processing (right before subsequent downsampling).
#             if level_idx > 0 and len(h) < 2:
#                 h.append(x)

#         res_block = self.mid
#         x = res_block(x)
        
#         h_idx = len(h) - 1
        
#         for level_idx in range(len(self.ups)):
#             upsample = self.ups[level_idx][0]
#             res_block = self.ups[level_idx][1]
            
#             x = upsample(x)
#             if level_idx < 2:
#                 x = torch.cat((x, h[h_idx]), dim=1)
#                 h_idx -= 1
            
#             x = res_block(x)
            
#         x = self.final_conv_block(x)
#         x = self.final_conv(x)

#         return x
    
