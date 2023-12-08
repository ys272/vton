from model import *

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
model_aux = Clothing_Autoencoder(channels=num_start_channels, init_dim=init_dim, level_dims=level_dims_aux, training_mode=False).to(c.DEVICE)
print(f'Total parameters in the aux model:  {sum(p.numel() for p in model_aux.parameters()):,}')

model_state = torch.load(os.path.join(c.MODEL_OUTPUT_PARAMS_DIR, '06-December-16:31_MIN_120K_medium_strided_USE_THIS.pth'))
model_aux.load_state_dict(model_state['model_aux_state_dict'])
model_aux.eval()
size = c.IMAGE_SIZE

'''
PARAMS
'''

base_image_size = c.IMAGE_SIZE
data_dir = '/home/yoni/Desktop/f/data/ready_datasets/vton_m_to_m_/'
target_dir = os.path.join(data_dir, 'AE')

for subdir in os.listdir(data_dir):
  print(f'Starting subdir {subdir}')
  data_subdir_path = os.path.join(data_dir, subdir)
  img_counter = 0
  clothing_arr = []
  sample_unique_string_ids = []
  for i,clothing_filename in enumerate(os.listdir(data_subdir_path)): 
    clothing_filename_no_suffix = clothing_filename.split('.')[0]
    if not clothing_filename_no_suffix.endswith('clothing'):
      continue
    clothing = torch.load(os.path.join(data_subdir_path, clothing_filename))
    clothing_arr.append(clothing)
    sample_unique_string_ids.append(clothing_filename_no_suffix)
  print(f'Containing {len(clothing_arr)} images')
  start_batch_idx = 0
  num_samples = len(clothing_arr)
  max_batch_size = 128
  while start_batch_idx < num_samples-1:
    end_batch_idx = min(start_batch_idx+128, num_samples)
    num_samples_batch = end_batch_idx - start_batch_idx
    clothing_batch = torch.stack(clothing_arr[start_batch_idx:end_batch_idx]).to(c.DEVICE).to(torch.bfloat16)
    with torch.no_grad():
      with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=c.USE_AMP):
        reconstructed_images = model_aux(clothing_batch)
    # reconstructed_images = torch.clamp(reconstructed_images, min=-1, max=1)
    # clothing_img_pred = (((reconstructed_images[0].to(dtype=torch.bfloat16).cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
    # cv2.imwrite(os.path.join('/home/yoni/Desktop', f'{sample_unique_string_ids[img_idx]}_{img_counter}_clothing_pred.jpg'), clothing_img_pred)
    for img_idx in range(len(reconstructed_images[0])):
      torch.save(reconstructed_images[0][img_idx].cpu(), os.path.join(target_dir, f'{sample_unique_string_ids[img_counter]}_0.pth'))
      torch.save(reconstructed_images[1][img_idx].cpu(), os.path.join(target_dir, f'{sample_unique_string_ids[img_counter]}_1.pth'))
      torch.save(reconstructed_images[2][img_idx].cpu(), os.path.join(target_dir, f'{sample_unique_string_ids[img_counter]}_2.pth'))
      img_counter += 1
      if img_counter%500==0:
        print(f'processing img #{img_counter}')
    start_batch_idx = end_batch_idx
  
