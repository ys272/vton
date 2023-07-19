import torch
import torch.nn.functional as F
import config as c
from tqdm import tqdm
import numpy as np


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


'''
TODO: The extreme leap of the final alpha (and correspondingly, beta) values of the cosine schedule seems suspiciously non smooth.
The final alphas / betas directly affect the noisy samples taken when t~=T.
Experiment with one of the following two options:
1. Using the alpha values of the linear beta schedule, while using the alpha_cumprod of the cosine beta schedule.
2. Just using the linear beta schedule, by adjusting the beta_end value such that alpha_cumprod only reaches 0 in a "smooth" 
fashion, at the final timestep, like the cosine schedule.
'''
betas = cosine_beta_schedule(c.NUM_TIMESTEPS)
# betas = linear_beta_schedule(c.timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def extract(a, t, x_shape):
    '''
    This will gather the values in array `a` at the indices within the `t` array, and reshape
    the resulting 1D array to e.g [batch_size, 1, 1, 1, 1] so it can be broadcasted with the downstream matrix.
    '''
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise



@torch.no_grad()
def p_sample_ddpm(model, x_t, t, t_index, clip_model_output=True):
    '''
    Predict x_t1 (x at timestep t-1).
    '''
    betas_t = extract(betas, t, x_t.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_t.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x_t.shape)
    
    model_output = model(x_t, t)
    if clip_model_output:
        model_output.clamp(c.MIN_NORMALIZED_VALUE, c.MAX_NORMALIZED_VALUE)

    if t_index == 0:
        # model_mean = sqrt_recip_alphas_t * (
        #     x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        # )
        
        # TODO!!!
        # Check whether this is better than the above (original). Here, we set 
        # sqrt_recip_alphas_t to 1.
        model_mean = x_t - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        return model_mean
    else:
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = extract(posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    

@torch.no_grad()
def p_sample_ddim(model, x_t:np.ndarray, t:int, t_index, clip_model_output:bool=True):
  '''
  Predict x_t1 (x at timestep t-1) deterministically (DDIM style, with eta = 0).
  '''
  model_output = model(x_t, t)
  if clip_model_output:
    model_output.clamp(c.MIN_NORMALIZED_VALUE, c.MAX_NORMALIZED_VALUE)
  alphas_cumprod_t = extract(
        alphas_cumprod, t, x_t.shape
  )
  alphas_cumprod_prev_t = extract(
        alphas_cumprod_prev, t, x_t.shape
  )
  betas_cumprod_t = 1 - alphas_cumprod_t
  betas_cumprod_prev_t = 1 - alphas_cumprod_prev_t
  
  if t_index == 1:
    # set alphas_cumprod_prev to 1, and betas_cumprod_prev_t to 0.
    alphas_cumprod_prev /= alphas_cumprod_prev 
    betas_cumprod_prev_t -= betas_cumprod_prev_t
      
  x_0_hat = (x_t - betas_cumprod_t.sqrt() * model_output) / alphas_cumprod_t.sqrt()
  x_t1 = alphas_cumprod_prev_t.sqrt() * x_0_hat + betas_cumprod_prev_t.sqrt() * model_output
  
  return x_t1

    
if c.REVERSE_DIFFUSION_SAMPLER == 'ddpm':
    reverse_sampler_func = p_sample_ddpm
elif c.REVERSE_DIFFUSION_SAMPLER == 'ddim':
    reverse_sampler_func = p_sample_ddim    

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device
    batch_size = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []
    for timestep in tqdm(reversed(range(0, c.NUM_TIMESTEPS)), desc='sampling loop time step', total=c.NUM_TIMESTEPS):
        img = reverse_sampler_func(model, img, torch.full((batch_size,), timestep, device=device, dtype=torch.long), timestep)
        imgs.append(img.cpu().numpy())
    return imgs


# import matplotlib.pyplot as plt
# plt.plot(alphas_cumprod, label='alpha bar')
# plt.plot(alphas, label='alpha')
# plt.plot(betas, label='beta')
# # plt.plot(posterior_variance, label='z')
# plt.legend()
# plt.show()
