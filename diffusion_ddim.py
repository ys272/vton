from torchvision.utils import save_image
import os
import cv2
import torch
import torch.nn.functional as F
import config as c
from tqdm import tqdm
import numpy as np
from functools import partial
from utils import denormalize_img, downsample_and_upsample_person, preprocess_s_person_output_for_m
import sys


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
    beta_end = 0.03
    return torch.linspace(beta_start, beta_end, timesteps)


'''
TODO: The extreme leap of the final alpha (and correspondingly, beta) values of the cosine schedule seems suspiciously non smooth.
The final alphas / betas directly affect the noisy samples taken when t~=T.
Experiment with one of the following two options:
1. Using the alpha values of the linear beta schedule, while using the alpha_cumprod of the cosine beta schedule.
2. Just using the linear beta schedule, by adjusting the beta_end value such that alpha_cumprod only reaches 0 in a "smooth" 
fashion, at the final timestep, like the cosine schedule.
'''
# betas = cosine_beta_schedule(c.NUM_TIMESTEPS)
betas = linear_beta_schedule(c.NUM_DIFFUSION_TIMESTEPS)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
# print(alphas_cumprod[:6], '\n',alphas_cumprod[-6:])

def extract(a, t, x_shape):
    '''
    This will gather the values in array `a` at the indices within the `t` array, and reshape
    the resulting 1D array to e.g [batch_size, 1, 1, 1, 1] so it can be broadcasted with the downstream matrix.
    '''
    batch_size = t.shape[0]
    # TODO: Can this call to "cpu" be removed?
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start) * c.NOISE_SCALING_FACTOR

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    # x_start = cv2.GaussianBlur(((x_start+1)*127.5).cpu().numpy().astype(np.uint8), (3,3), sigmaX=0.5)
    # x_start = torch.tensor((x_start / 127.5) -1, device=c.DEVICE)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    # return (sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise).clamp(c.MIN_NORMALIZED_VALUE, c.MAX_NORMALIZED_VALUE)


@torch.no_grad()
def p_sample_ddpm(model, x_t, t, t_index, clip_model_output=True):
    '''
    Predict x_t1 (x at timestep t-1).
    TODO: If using ddpm, note that it's unclear where/how to apply clamping/clipping, if requested.
    For now, use ddim with eta=1, if you want to simulate ddpm.
    '''
    sys.exit('ddpm needs to be refactored to support conditional sampling')
    betas_t = extract(betas, t, x_t.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_t.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x_t.shape)
    
    model_output = model(x_t, t)
    if t_index == 0:
        # model_mean = sqrt_recip_alphas_t * (
        #     x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        # )
        # TODO!!!
        # Check whether this is better than the above (original). Here, we set 
        # sqrt_recip_alphas_t to 1.
        model_mean = x_t - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        if clip_model_output:
            model_mean = model_mean.clamp(c.MIN_NORMALIZED_VALUE, c.MAX_NORMALIZED_VALUE)
        return model_mean
    else:
        model_mean =  x_t - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        posterior_variance_t = extract(posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t) * c.NOISE_SCALING_FACTOR

        return sqrt_recip_alphas_t * model_mean + torch.sqrt(posterior_variance_t) * noise
    

@torch.no_grad()
def p_sample_ddim(model_main, model_aux, inputs, x_t:np.ndarray, cross_attns:np.ndarray, t:int, t_index, clip_model_output:bool=True, eta=1, eval_mode=False, base_image_size='s', add_downsample_noise=False):
  '''
  Predict x_t1 (x at timestep t-1) using DDIM.
  If eta=0, it's deterministic.
  If eta=1, it's DDPM.
  Values in between 0 and 1 are an interpolation between DDIM and DDPM.
  '''
  
  clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, _, _, noise_amount_clothing, noise_amount_masked = inputs
  with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=c.USE_AMP):
    if c.USE_CLASSIFIER_FREE_GUIDANCE and eval_mode:
        cross_attns = model_aux(clothing_aug, pose_vector, noise_amount_clothing, t)
        x_t_and_masked_aug = torch.cat((x_t, masked_aug, pose_matrix, mask_coords.to(clothing_aug.dtype).unsqueeze(1)), dim=1)  
        model_output_conditional = model_main(x_t_and_masked_aug, pose_vector, noise_amount_masked, t, cross_attns=cross_attns)
        
        level_dims_aux = c.MODELS_PARAMS[base_image_size][1]
        cross_attns = [torch.zeros((clothing_aug.shape[0], level_dims_aux[-1], 16, 11), device=c.DEVICE), torch.zeros((clothing_aug.shape[0], level_dims_aux[-2], 32, 22), device=c.DEVICE)]
        mask_coords = torch.zeros_like(mask_coords)
        masked_aug = torch.zeros_like(masked_aug)
        pose_vector = torch.zeros_like(pose_vector)
        x_t_and_masked_aug = torch.cat((x_t, masked_aug, pose_matrix, mask_coords.to(clothing_aug.dtype).unsqueeze(1)), dim=1)  
        model_output_not_conditional = model_main(x_t_and_masked_aug, pose_vector, noise_amount_masked, t, cross_attns=cross_attns)
        
        # model_output = torch.lerp(model_output_not_conditional, model_output_conditional, 2)
        model_output = 1.25 * model_output_conditional - 0.25 * model_output_not_conditional
    else:
        cross_attns = model_aux(clothing_aug, pose_vector, noise_amount_clothing, t)
        x_t_and_masked_aug = torch.cat((x_t, masked_aug, pose_matrix, mask_coords.to(clothing_aug.dtype).unsqueeze(1)), dim=1)
        model_output = model_main(x_t_and_masked_aug, pose_vector, noise_amount_masked, t, cross_attns=cross_attns)
  
  alphas_cumprod_t = extract(alphas_cumprod, t, x_t.shape)
  alphas_cumprod_prev_t = extract(alphas_cumprod_prev, t, x_t.shape)
  betas_cumprod_t = 1 - alphas_cumprod_t
  betas_cumprod_prev_t = 1 - alphas_cumprod_prev_t

  if t_index == 0:
    # set alphas_cumprod_prev to 1, and betas_cumprod_prev_t to 0.
    alphas_cumprod_prev_t /= alphas_cumprod_prev_t
    betas_cumprod_prev_t -= betas_cumprod_prev_t
    
  sigma = eta * ((betas_cumprod_prev_t / betas_cumprod_t) * (1 - (alphas_cumprod_t / alphas_cumprod_prev_t))).sqrt()
  noise = torch.randn_like(x_t) * c.NOISE_SCALING_FACTOR
  
  x_0_hat = (x_t - betas_cumprod_t.sqrt() * model_output) / alphas_cumprod_t.sqrt()
#   print(t_index, 'before', torch.min(x_0_hat).item(), torch.max(x_0_hat).item(), eta
  if clip_model_output:
    x_0_hat = x_0_hat.clamp(c.MIN_NORMALIZED_VALUE, c.MAX_NORMALIZED_VALUE)
#   print(t_index, 'after', torch.min(x_0_hat).item(), torch.max(x_0_hat).item(), eta)
  x_t1 = alphas_cumprod_prev_t.sqrt() * x_0_hat + (betas_cumprod_prev_t - sigma**2).sqrt() * model_output + sigma * noise
  
  return x_t1, cross_attns


@torch.no_grad()
def p_sample_loop(model_main, model_aux, inputs, shape, base_image_size, sampler=c.REVERSE_DIFFUSION_SAMPLER, clip_model_output=True, eta=None, eval_mode=False, add_downsample_noise=False):
    if sampler == 'ddpm':
        reverse_sampler_func = p_sample_ddpm
    elif sampler == 'ddim':
        if eta is not None:
            reverse_sampler_func = partial(p_sample_ddim, eta=eta, eval_mode=eval_mode, base_image_size=base_image_size, add_downsample_noise=add_downsample_noise)
        else:
            reverse_sampler_func = partial(p_sample_ddim, eval_mode=eval_mode, base_image_size=base_image_size, add_downsample_noise=add_downsample_noise)
    batch_size = shape[0]
    # start from pure noise (for each example in the batch)
    img_initially_noise = torch.randn(shape, device=c.DEVICE) * c.NOISE_SCALING_FACTOR
    cross_attns = None
    imgs = []
    for timestep in tqdm(reversed(range(0, c.NUM_DIFFUSION_TIMESTEPS)), desc='sampling loop time step', total=c.NUM_DIFFUSION_TIMESTEPS):
        img_initially_noise, cross_attns = reverse_sampler_func(model_main, model_aux, inputs, img_initially_noise, cross_attns, torch.full((batch_size,), timestep, device=c.DEVICE, dtype=torch.long), timestep, clip_model_output=clip_model_output)
        imgs.append(img_initially_noise)
    return imgs


def show_example_noise_sequence(imgs):
    for img_idx,img in enumerate(imgs):
        for t_idx in range(c.NUM_DIFFUSION_TIMESTEPS):
            t = torch.tensor([t_idx]).cuda()
            noised_img = q_sample(img, t)

            noised_img_save_path = os.path.join('/home/yoni/Desktop/f/other/debugging/noising_examples', f'{img_idx}_{t_idx}.png')
            noised_img = (noised_img.cpu().numpy() + 1) * 127.5
            cv2.imwrite(noised_img_save_path, noised_img[::-1].transpose(1,2,0))
        cv2.imwrite(f'/home/yoni/Desktop/f/other/debugging/noising_examples/{img_idx}_0_nonoise.png', ((img.cpu().numpy() + 1) * 127.5).astype(np.uint8)[::-1].transpose(1,2,0))


def call_sampler_simple(model_main, model_aux, inputs, shape, base_image_size, sampler=c.REVERSE_DIFFUSION_SAMPLER, clip_model_output=True, show_all=False, eta=None, eval_mode=False, original_indices=None):
    if not original_indices:
        original_indices = list(range(shape[0]))
    img_sequences = p_sample_loop(model_main, model_aux, inputs, shape, base_image_size=base_image_size, sampler=c.REVERSE_DIFFUSION_SAMPLER, clip_model_output=True, eta=None, eval_mode=eval_mode)
    clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = inputs
    if not show_all:
        for img_idx,img in enumerate(img_sequences[-1]):
            save_idx = original_indices[img_idx]
            img = denormalize_img(img)
            save_image(img, os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{sample_unique_string_id[img_idx]}_{save_idx}_PRED.png'), nrow = 4//2)
            masked_img = (((masked_aug[img_idx].to(dtype=torch.float16).cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
            person_img = (((person[img_idx].to(dtype=torch.float16).cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
            clothing_img = (((clothing_aug[img_idx].to(dtype=torch.float16).cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
            cv2.imwrite(os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{sample_unique_string_id[img_idx]}_{img_idx}_masked.png'), masked_img)
            cv2.imwrite(os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{sample_unique_string_id[img_idx]}_{img_idx}_person.png'), person_img)
            cv2.imwrite(os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{sample_unique_string_id[img_idx]}_{save_idx}_clothing.png'), clothing_img)
    else:
        for img_idx in range(shape[0]):
            save_idx = original_indices[img_idx]
            for t_idx,imgs in enumerate(img_sequences):
                img = denormalize_img(imgs[img_idx].squeeze(0))
                save_image(img, os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{sample_unique_string_id}_{save_idx}_{c.NUM_DIFFUSION_TIMESTEPS-t_idx-1}_PRED.png'), nrow = 4//2)
            masked_img = (((masked_aug[img_idx].to(dtype=torch.float16).cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
            person_img = (((person[img_idx].to(dtype=torch.float16).cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
            clothing_img = (((clothing_aug[img_idx].to(dtype=torch.float16).cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
            cv2.imwrite(os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{sample_unique_string_id[img_idx]}_{save_idx}_masked.png'), masked_img)
            cv2.imwrite(os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{sample_unique_string_id[img_idx]}_{save_idx}_person.png'), person_img)
            cv2.imwrite(os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{sample_unique_string_id[img_idx]}_{save_idx}_clothing.png'), clothing_img)
