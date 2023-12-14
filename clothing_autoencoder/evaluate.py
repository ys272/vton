from model import *
import config as c
import os
import cv2
import numpy as np


if c.USE_AMP:
    if c.USE_BFLOAT16:
        MODEL_DTYPE = torch.bfloat16
    else:
        MODEL_DTYPE = torch.float16
else:
    MODEL_DTYPE = torch.float32
    

img_height = c.VTON_RESOLUTION[c.IMAGE_SIZE][0]
img_width = c.VTON_RESOLUTION[c.IMAGE_SIZE][1]
init_dim = c.MODELS_INIT_DIM

level_dims_main = c.MODELS_PARAMS[c.IMAGE_SIZE][0]
level_dims_aux = c.MODELS_PARAMS[c.IMAGE_SIZE][1]
level_attentions = c.MODELS_PARAMS[c.IMAGE_SIZE][2]
level_repetitions_main = c.MODELS_PARAMS[c.IMAGE_SIZE][3]
level_repetitions_aux = c.MODELS_PARAMS[c.IMAGE_SIZE][4]
num_start_channels = 3
model_aux = Clothing_Autoencoder(channels=num_start_channels, init_dim=init_dim, level_dims=level_dims_aux).to(c.DEVICE)
print(f'Total parameters in the aux model:  {sum(p.numel() for p in model_aux.parameters()):,}')

model_state = torch.load('/home/yoni/Desktop/model weights/06-December-16:31_MIN_120K_medium_strided_USE_THIS_AUTOENCODER.pth')
model_aux.load_state_dict(model_state['model_aux_state_dict'])
model_aux.eval()
size = c.IMAGE_SIZE

'''
PARAMS
'''

base_image_size = c.IMAGE_SIZE

unaligned_test_dataset_dir = '/home/yoni/Desktop/f/test/ready_data/'
clothing_augs = []
noise_amount_clothings = []
sample_unique_string_ids = []

for clothing_filename in os.listdir(os.path.join(unaligned_test_dataset_dir,'clothing', 'm')): 
  clothing = cv2.imread(os.path.join(unaligned_test_dataset_dir,'clothing', 'm', clothing_filename))
  clothing = (clothing / 127.5) - 1
  clothing = clothing.transpose(2,0,1)
  clothing = np.copy(clothing.astype(np.float16)[::-1])
  clothing = torch.from_numpy(clothing)
  
  clothing_augs.append(clothing)
  sample_unique_string_ids.append(clothing_filename)
  
num_samples = len(clothing_augs)
start_batch_idx = 0
max_batch_size = 8
img_counter = 0
while start_batch_idx < num_samples-1:
  end_batch_idx = min(start_batch_idx+8, num_samples-1)
  num_samples_batch = end_batch_idx - start_batch_idx
  
  clothing_augs_batch = torch.stack(clothing_augs[start_batch_idx:end_batch_idx]).cuda().to(torch.bfloat16)
  with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=c.USE_AMP):
      reconstructed_images  = model_aux(clothing_augs_batch)
  reconstructed_images = torch.clamp(reconstructed_images, min=-1, max=1)
  for img_idx in range(len(reconstructed_images)):
    clothing_img_pred = (((reconstructed_images[img_idx].to(dtype=torch.float16).cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
    clothing_img_orig = (((clothing_augs_batch[img_idx].to(dtype=torch.float16).cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
    cv2.imwrite(os.path.join('/home/yoni/Desktop/f/other/debugging/d', f'{sample_unique_string_ids[img_idx]}_{img_counter}_clothing_pred.jpg'), clothing_img_pred)
    cv2.imwrite(os.path.join('/home/yoni/Desktop/f/other/debugging/d', f'{sample_unique_string_ids[img_idx]}_{img_counter}_clothing_orig.jpg'), clothing_img_orig)
    img_counter += 1
  start_batch_idx = end_batch_idx
