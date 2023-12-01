from torch.utils.data import DataLoader
from model import *
from diffusion_ddim import call_sampler_simple
from diffusion_karras import call_sampler_simple_karras
from datasets import CustomDataset, process_keypoints


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
    
# model_main = Unet_Person_Masked(channels=19, init_dim=init_dim, level_dims=level_dims_main, level_dims_cross_attn=level_dims_aux, level_attentions=level_attentions,level_repetitions = level_repetitions_main,).to(c.DEVICE)
# model_aux = Unet_Clothing(channels=3, init_dim=init_dim, level_dims=level_dims_aux,level_repetitions=level_repetitions_aux,).to(c.DEVICE)
model_main = Unet_Person_Masked(channels=19, init_dim=init_dim, level_dims=level_dims_main, level_dims_cross_attn=level_dims_aux, level_attentions=level_attentions,level_repetitions = level_repetitions_main,base_image_size=c.IMAGE_SIZE).to(c.DEVICE)
model_aux = Unet_Clothing(channels=3, init_dim=init_dim, level_dims=level_dims_aux,level_repetitions=level_repetitions_aux,).to(c.DEVICE)
print(f'Total parameters in the main model: {sum(p.numel() for p in model_main.parameters()):,}')
print(f'Total parameters in the aux model:  {sum(p.numel() for p in model_aux.parameters()):,}')

model_state = torch.load(os.path.join(c.MODEL_OUTPUT_PARAMS_DIR, '23-November-23:46_6908176_normal_loss_0.020.pth'))
model_main.load_state_dict(model_state['model_ema_main_state_dict'])
model_aux.load_state_dict(model_state['model_ema_aux_state_dict'])
model_main.eval()
model_aux.eval()
size = c.IMAGE_SIZE
test_dataloader = None #torch.load(f'/home/yoni/Desktop/f/data/ready_datasets/test_dataloader_{size}.pth')

'''
PARAMS
'''

shuffle = False
unaligned_test_dataset = True
base_image_size = c.IMAGE_SIZE

if not unaligned_test_dataset:
  if shuffle:
    test_dataloader_w_shuffle = DataLoader(test_dataloader.dataset, batch_size=test_dataloader.batch_size, shuffle=True)
    clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = next(iter(test_dataloader_w_shuffle))
  else:
    test_dataloader = iter(test_dataloader)
    clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = next(test_dataloader)
    clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = next(test_dataloader)
    clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = next(test_dataloader)
    clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = next(test_dataloader)
    clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = next(test_dataloader)
    clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = next(test_dataloader)
    clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked = next(test_dataloader)

  num_eval_samples = min(8, clothing_aug.shape[0])
  if not c.USE_BFLOAT16:
    inputs = [clothing_aug[:num_eval_samples].cuda().float(), mask_coords[:num_eval_samples].cuda(), masked_aug[:num_eval_samples].cuda().float(), person[:num_eval_samples].cuda().float(), pose_vector[:num_eval_samples].cuda().float(), pose_matrix[:num_eval_samples].cuda().float(), sample_original_string_id, sample_unique_string_id, noise_amount_clothing[:num_eval_samples].cuda().float(), noise_amount_masked[:num_eval_samples].cuda().float()]
  else:
    inputs = [clothing_aug[:num_eval_samples].cuda(), mask_coords[:num_eval_samples].cuda(), masked_aug[:num_eval_samples].cuda(), person[:num_eval_samples].cuda(), pose_vector[:num_eval_samples].cuda(), pose_matrix[:num_eval_samples].cuda(), sample_original_string_id, sample_unique_string_id, noise_amount_clothing[:num_eval_samples].cuda(), noise_amount_masked[:num_eval_samples].cuda()]
  if c.REVERSE_DIFFUSION_SAMPLER == 'ddim':
    imgs = call_sampler_simple(model_main, model_aux, inputs, shape=(num_eval_samples, 3, img_height, img_width), base_image_size=base_image_size, sampler='ddim', clip_model_output=True, show_all=False, eta=1, eval_mode=True)
  else:
    imgs = call_sampler_simple_karras(model_main, model_aux, inputs, sampler='euler_ancestral', steps=250, sigma_max=c.KARRAS_SIGMA_MAX, clip_model_output=True, show_all=False)
else:
  unaligned_test_dataset_dir = '/home/yoni/Desktop/f/test/ready_data/'
  for person_filename in os.listdir(os.path.join(unaligned_test_dataset_dir,'person_original', 'm')):
    person = cv2.imread(os.path.join(unaligned_test_dataset_dir,'person_original', 'm', person_filename))
    person = (person / 127.5) - 1
    person = person.transpose(2,0,1)
    person = np.copy(person.astype(np.float16)[::-1])
    person = torch.from_numpy(person)
    
    masked = cv2.imread(os.path.join(unaligned_test_dataset_dir,'person_with_masked_clothing', 'm', person_filename))
    masked = (masked / 127.5) - 1
    masked = masked.transpose(2,0,1)
    masked = np.copy(masked.astype(np.float16)[::-1])
    masked = torch.from_numpy(masked)
    
    mask_coords = torch.from_numpy(np.load(os.path.join(unaligned_test_dataset_dir,'mask_coordinates', 'm', person_filename.split('.')[0] + '.npy')))
    with open(os.path.join(unaligned_test_dataset_dir, 'pose_keypoints', 'm', person_filename.split('.')[0] + '.txt'), 'r') as f:
      pose_vector = process_keypoints(eval(f.readlines()[0]))
    num_needed_keypoint_dims = 12
    pose_matrix = torch.zeros((num_needed_keypoint_dims, img_height, img_width), dtype=MODEL_DTYPE)
    # The vector flattened the pairs to a single 1D list, so the first 5 keypoints pairs now take 10 elements in total.
    for p_idx in range(10, len(pose_vector), 2):
        # We flip the order of the keypoints because pytorch and tensorflow (where the keypoints come from) use a different axis ordering system.
        y = pose_vector[p_idx]
        x = pose_vector[p_idx+1]
        if x==0 and y==0:
            continue
        x = torch.round(x * img_height)
        y = torch.round(y * img_width)
        pose_matrix[int((p_idx - 10)/2)][int(min(img_height - 1, x)), int(min(img_width - 1, y))] = 1
        
    noise_amount_masked = 0.01
    noise_tensor = torch.randn_like(masked)
    masked_aug = masked * (1 - noise_amount_masked) + noise_tensor * noise_amount_masked
    masked_aug[:, mask_coords] = masked[:, mask_coords]
    
    clothing_augs = []
    noise_amount_clothings = []
    sample_unique_string_ids = []

    for clothing_filename in os.listdir(os.path.join(unaligned_test_dataset_dir,'clothing', 'm')): 
      clothing = cv2.imread(os.path.join(unaligned_test_dataset_dir,'clothing', 'm', clothing_filename))
      clothing = (clothing / 127.5) - 1
      clothing = clothing.transpose(2,0,1)
      clothing = np.copy(clothing.astype(np.float16)[::-1])
      clothing = torch.from_numpy(clothing)
      
      noise_amount_clothing = 0#0.01
      noise_tensor = torch.randn_like(clothing)
      clothing_aug = clothing * (1 - noise_amount_clothing) + noise_tensor * noise_amount_clothing
      clothing_augs.append(clothing_aug)
      noise_amount_clothings.append(noise_amount_clothing)
      sample_unique_string_ids.append(person_filename + '_' + clothing_filename)
      
    num_samples = len(clothing_augs)
    start_batch_idx = 0
    max_batch_size = 8
    while start_batch_idx < num_samples-1:
      end_batch_idx = min(start_batch_idx+8, num_samples-1)
      num_samples_batch = end_batch_idx - start_batch_idx
      
      clothing_augs_batch = torch.stack(clothing_augs[start_batch_idx:end_batch_idx]).cuda().to(torch.bfloat16)
      mask_coords_batch = torch.stack([mask_coords.cuda().clone() for _ in range(num_samples_batch)]).to(torch.bfloat16)
      masked_aug_batch = torch.stack([masked_aug.cuda().clone() for _ in range(num_samples_batch)]).to(torch.bfloat16)
      person_batch = torch.stack([person.cuda().clone() for _ in range(num_samples_batch)]).to(torch.bfloat16)
      pose_vector_batch = torch.stack([pose_vector.cuda().clone() for _ in range(num_samples_batch)]).to(torch.bfloat16)
      pose_matrix_batch = torch.stack([pose_matrix.cuda().clone() for _ in range(num_samples_batch)]).to(torch.bfloat16)
      noise_amount_clothing_batch = torch.tensor([noise_amount_clothing] * num_samples_batch, device='cuda').to(torch.bfloat16)
      noise_amount_masked_batch = torch.tensor([noise_amount_masked] * num_samples_batch, device='cuda').to(torch.bfloat16)
      
      inputs = [clothing_augs_batch, mask_coords_batch, masked_aug_batch, person_batch, pose_vector_batch, pose_matrix_batch, sample_unique_string_ids, sample_unique_string_ids, noise_amount_clothing_batch, noise_amount_masked_batch]
      imgs = call_sampler_simple(model_main, model_aux, inputs, shape=(num_samples_batch, 3, img_height, img_width), base_image_size=base_image_size, sampler='ddim', clip_model_output=True, show_all=False, eta=1, eval_mode=False, original_indices=list(range(start_batch_idx, start_batch_idx+num_samples_batch)))
      start_batch_idx = end_batch_idx
