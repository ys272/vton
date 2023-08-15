from torch.utils.data import DataLoader
from model import *
from diffusion_ddim import call_sampler_simple
from diffusion_karras import call_sampler_simple_karras
from datasets import CustomDataset


img_height = c.VTON_RESOLUTION[c.IMAGE_SIZE][0]
img_width = c.VTON_RESOLUTION[c.IMAGE_SIZE][1]
init_dim = 128

level_dims_main = (128, 512, 512)
level_dims_aux = (128, 512, 512)
level_attentions = (False, True)
level_repetitions_main = (2,4,4)
level_repetitions_aux = (2,4,4)
    
model_main = Unet_Person_Masked(channels=6, init_dim=init_dim, level_dims=level_dims_main, level_dims_cross_attn=level_dims_aux, level_attentions=level_attentions,level_repetitions = level_repetitions_main,).to(c.DEVICE)
model_aux = Unet_Clothing(channels=3, init_dim=init_dim, level_dims=level_dims_aux,level_repetitions=level_repetitions_aux,).to(c.DEVICE)
print(f'Total parameters in the main model: {sum(p.numel() for p in model_main.parameters()):,}')
print(f'Total parameters in the aux model:  {sum(p.numel() for p in model_aux.parameters()):,}')

model_state = torch.load(os.path.join(c.MODEL_OUTPUT_PARAMS_DIR, '14-August-ddim_e-4,batch size grow.pth'))
model_main.load_state_dict(model_state['model_main_state_dict'])
model_aux.load_state_dict(model_state['model_aux_state_dict'])
model_main.eval()
model_aux.eval()
size = c.IMAGE_SIZE
test_dataloader = torch.load(f'/home/yoni/Desktop/f/data/ready_datasets/test_dataloader_{size}.pth')

'''
PARAMS
'''

shuffle = True


if shuffle:
  test_dataloader_w_shuffle = DataLoader(test_dataloader.dataset, batch_size=test_dataloader.batch_size, shuffle=True)
  clothing_aug, mask_coords, masked_aug, person, pose, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = next(iter(test_dataloader_w_shuffle))
else:
  clothing_aug, mask_coords, masked_aug, person, pose, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = next(iter(test_dataloader))

num_eval_samples = min(8, clothing_aug.shape[0])
inputs = [clothing_aug[:num_eval_samples].cuda().float(), mask_coords[:num_eval_samples].cuda(), masked_aug[:num_eval_samples].cuda().float(), person[:num_eval_samples].cuda().float(), pose[:num_eval_samples].cuda().float(), sample_original_string_id, sample_unique_string_id, noise_amount_clothing[:num_eval_samples].cuda().float(), noise_amount_masked[:num_eval_samples].cuda().float()]
if c.REVERSE_DIFFUSION_SAMPLER == 'ddim':
  imgs = call_sampler_simple(model_main, model_aux, inputs, shape=(num_eval_samples, 3, img_height, img_width), sampler='ddim', clip_model_output=True, show_all=True, eta=1)
else:
  imgs = call_sampler_simple_karras(model_main, model_aux, inputs, sampler='euler_ancestral', steps=250, sigma_max=c.KARRAS_SIGMA_MAX, clip_model_output=True, show_all=True)

print('')