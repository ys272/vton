from scipy import integrate
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
import torch, os, cv2
from utils import denormalize_img
import config as c


'''
Karras
'''
sig_data = 0.7

def q_sample_karras(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0) * c.NOISE_SCALING_FACTOR
    sig = t.reshape(-1,1,1,1)
    c_skip,c_out,c_in = scalings_karras(sig)
    noised_input = x0 + noise*sig
    target = (x0 -c_skip*noised_input) / c_out
    return noised_input*c_in, target


def scalings_karras(sig):
    totvar = sig**2 + sig_data**2
    # c_skip, c_out, c_in
    return sig_data**2/totvar, sig*sig_data/totvar.sqrt(), 1/totvar.sqrt()


def sigmas_karras(n, sigma_min=0.01, sigma_max=c.KARRAS_SIGMA_MAX, rho=7.):
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min**(1/rho)
    max_inv_rho = sigma_max**(1/rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho-max_inv_rho))**rho
    return torch.cat([sigmas, torch.tensor([0.])]).to(c.DEVICE)


def denoise_karras(model_main, model_aux, x, sig, masked_aug, clothing_aug, pose, noise_amount_masked, noise_amount_clothing):
    c_skip,c_out,c_in = scalings_karras(sig)
    t = torch.full((x.shape[0],), sig, device=c.DEVICE)
    x_t_and_masked_aug = torch.cat((x*c_in, masked_aug), dim=1)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=c.USE_AMP):
        cross_attns = model_aux(clothing_aug, pose, noise_amount_clothing, t)
        model_output = model_main(x_t_and_masked_aug, pose, noise_amount_masked, t, cross_attns) * c_out + x * c_skip

    return model_output


@torch.no_grad()
def sample_euler_karras(x, sigs, i, model_main, model_aux, masked_aug, clothing_aug, pose, noise_amount_masked, noise_amount_clothing):
    sig,sig2 = sigs[i],sigs[i+1]
    denoised = denoise_karras(model_main, model_aux, x, sig, masked_aug, clothing_aug, pose, noise_amount_masked, noise_amount_clothing)
    return x + (x-denoised)/sig*(sig2-sig)


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    if not eta: return sigma_to, 0.
    var_to,var_from = sigma_to**2,sigma_from**2
    sigma_up = min(sigma_to, eta * (var_to * (var_from-var_to)/var_from)**0.5)
    return (var_to-sigma_up**2)**0.5, sigma_up
     

@torch.no_grad()
def sample_euler_ancestral_karras(x, sigs, i, model_main, model_aux, masked_aug, clothing_aug, pose, noise_amount_masked, noise_amount_clothing, eta=1):
    sig,sig2 = sigs[i],sigs[i+1]
    denoised = denoise_karras(model_main, model_aux, x, sig, masked_aug, clothing_aug, pose, noise_amount_masked, noise_amount_clothing)
    sigma_down,sigma_up = get_ancestral_step(sig, sig2, eta=eta)
    x = x + (x-denoised)/sig*(sigma_down-sig)
    return x + torch.randn_like(x)*sigma_up


def linear_multistep_coeff(order, t, i, j):
    if order-1 > i: raise ValueError(f'Order {order} too high for step {i}')
    def fn(tau):
        prod = 1.
        for k in range(order):
            if j == k: continue
            prod *= (tau-t[i-k]) / (t[i-j]-t[i-k])
        return prod
    return integrate.quad(fn, t[i], t[i+1], epsrel=1e-4)[0]


@torch.no_grad()
def sample_lms(model, num_samples=4, steps=100, order=4, sigma_max=c.KARRAS_SIGMA_MAX):
    preds = []
    x = torch.randn((num_samples,1,28,28), device=c.DEVICE)*sigma_max
    sigs = sigmas_karras(steps, sigma_max=sigma_max)
    ds = []
    for i in tqdm(range(len(sigs)-1)):
        sig = sigs[i]
        denoised = denoise_karras(model, x, sig)
        d = (x-denoised)/sig
        ds.append(d)
        if len(ds) > order: ds.pop(0)
        cur_order = min(i+1, order)
        coeffs = [linear_multistep_coeff(cur_order, sigs, i, j) for j in range(cur_order)]
        x = x + sum(coeff*d for coeff, d in zip(coeffs, reversed(ds)))
        preds.append(x)
    return preds


def p_sample_loop_karras(sampler, model_main, model_aux, inputs, steps=100, sigma_max=c.KARRAS_SIGMA_MAX, **kwargs):
    preds = []
    x = torch.randn((inputs[0].shape[0],inputs[0].shape[1],inputs[0].shape[2],inputs[0].shape[3]), device=c.DEVICE)*sigma_max
    sigs = sigmas_karras(steps, sigma_max=sigma_max)
    clothing_aug, mask_coords, masked_aug, person, pose, _, _, noise_amount_clothing, noise_amount_masked = inputs
    # with torch.cuda.amp.autocast(dtype=torch.float16):
    #     cross_attns = model_aux(clothing_aug, pose, noise_amount_clothing)
    # for i in tqdm(range(len(sigs)-1)):
    #     x = sampler(x, sigs, i, model_main, masked_aug, pose, noise_amount_masked, cross_attns, **kwargs)
    #     preds.append(x)
    for i in tqdm(range(len(sigs)-1)):
        x = sampler(x, sigs, i, model_main, model_aux, masked_aug, clothing_aug, pose, noise_amount_masked, noise_amount_clothing)
        preds.append(x)
    return preds
  

def show_example_noise_sequence_karras(imgs, steps=c.NUM_TIMESTEPS, sigma_max=c.KARRAS_SIGMA_MAX, rho=7.0):
    for img_idx,img in enumerate(imgs):
        # TODO: This value of rho might be more suitable for our purposes since it spends less time
        # in the extremely high noise areas.
        # noise_levels = sigmas_karras(steps, rho=15)
        noise_levels = sigmas_karras(steps, sigma_max=sigma_max, rho=rho)
        for t_idx,t in enumerate(reversed(noise_levels)):
            t = torch.tensor([t]).cuda()
            noised_img,_ = q_sample_karras(img,t)
            noised_img_save_path = os.path.join('/home/yoni/Desktop/f/other/debugging/noising_examples', f'{img_idx}_{t_idx}.png')
            noised_img = denormalize_img(noised_img.cpu().numpy()) * 255
            cv2.imwrite(noised_img_save_path, noised_img.squeeze()[::-1].transpose(1,2,0))
            

def call_sampler_simple_karras(model_main, model_aux, inputs, sampler='euler_ancestral', steps=100, sigma_max=c.KARRAS_SIGMA_MAX, clip_model_output=True, show_all=False):
    if sampler == 'euler':
        img_sequences = p_sample_loop_karras(sample_euler_karras, model_main, model_aux, inputs, steps=steps, sigma_max=sigma_max)
    elif sampler == 'euler_ancestral':
        img_sequences = p_sample_loop_karras(sample_euler_ancestral_karras, model_main, model_aux, inputs, steps=steps, sigma_max=sigma_max)
    elif sampler == 'lms':
        pass
        # img_sequences = sample_lms(model, num_samples=num_samples, steps=steps, order=4, sigma_max=sigma_max)
    clothing_aug, mask_coords, masked_aug, person, pose, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = inputs
    if not show_all:
        for img_idx,img in enumerate(img_sequences[-1]):
            img = denormalize_img(img)
            save_image(img, os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{img_idx}_PRED.png'), nrow = 4//2)
            masked_img = (((masked_aug[img_idx].cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
            person_img = (((person[img_idx].cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
            clothing_img = (((clothing_aug[img_idx].cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
            cv2.imwrite(os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{img_idx}_masked.png'), masked_img)
            cv2.imwrite(os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{img_idx}_person.png'), person_img)
            cv2.imwrite(os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{img_idx}_clothing.png'), clothing_img)
    else:
        for img_idx in range(inputs[0].shape[0]):
            for t_idx,imgs in enumerate(img_sequences):
                img = denormalize_img(imgs[img_idx].squeeze(0))
                save_image(img, os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{img_idx}_{steps-t_idx-1}.png'), nrow = 4//2)
            masked_img = (((masked_aug[img_idx].cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
            person_img = (((person[img_idx].cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
            clothing_img = (((clothing_aug[img_idx].cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
            cv2.imwrite(os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{img_idx}_masked.png'), masked_img)
            cv2.imwrite(os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{img_idx}_person.png'), person_img)
            cv2.imwrite(os.path.join('/home/yoni/Desktop/f/other/debugging/denoising_examples', f'{img_idx}_clothing.png'), clothing_img)
    return img_sequences



# import matplotlib.pyplot as plt
# sigmas15 = sigmas_karras(100, rho=15)
# sigmas9 = sigmas_karras(100, rho=9)
# sigmas7 = sigmas_karras(100, rho=7)
# sigmas5 = sigmas_karras(100, rho=5)
# sigmas3 = sigmas_karras(100, rho=3)
# plt.plot(sigmas15.cpu(), label='sigmas15')
# plt.plot(sigmas9.cpu(), label='sigmas9')
# plt.plot(sigmas7.cpu(), label='sigmas7')
# plt.plot(sigmas5.cpu(), label='sigmas5')
# plt.plot(sigmas3.cpu(), label='sigmas3')
# plt.legend()
# plt.show()
