import os
import numpy as np
import cv2
from tqdm import tqdm
import sys
from utils import resize_img
import pickle
import config as c
from data_preprocessing_vton.pose import PoseModel
from data_preprocessing_vton.schp import extract_person_without_clothing
from config import VTON_RESOLUTION
from random import random, uniform
from typing import Tuple, List
import torch.nn.functional as F
import torch


class GaussianBlur():
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, img, sigma):
        # Convert the image tensor to a PyTorch tensor
        # img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = torch.tensor(img).unsqueeze(0).float()
        # Generate a random sigma value
        sigma = torch.tensor(sigma)
        # Apply Gaussian blur using the specified kernel size and sigma
        img_blurred = self.apply_gaussian_blur(img_tensor, self.kernel_size, sigma)
        # Convert the blurred tensor back to an image tensor with shape H,W,3
        # blurred_image = img_blurred.squeeze(0).permute(1, 2, 0)
        blurred_image = img_blurred.squeeze(0)
        return blurred_image

    def apply_gaussian_blur(self, img, kernel_size, sigma):
        # Calculate padding required for 'same' convolution
        padding = kernel_size // 2
        # Create separate kernel tensors for each channel
        kernel_tensors = [self.create_gaussian_kernel(kernel_size, sigma) for _ in range(img.size(1))]
        # Perform 2D convolution on each channel independently
        img_blurred_channels = [F.conv2d(img[:, i:i+1], kernel_tensors[i], padding=padding) for i in range(img.size(1))]
        # Concatenate the blurred channels back into an RGB image
        img_blurred = torch.cat(img_blurred_channels, dim=1)
        return img_blurred

    def create_gaussian_kernel(self, kernel_size, sigma):
        # Create the 1D Gaussian kernel
        kernel_range = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (kernel_range / sigma)**2)
        kernel /= kernel.sum()
        # Extend the 1D kernel to 2D
        kernel_2d = torch.outer(kernel, kernel).unsqueeze(0).unsqueeze(0)
        return kernel_2d

'''
# Example usage:
img = torch.from_numpy(cv2.imread('/home/yoni/Desktop/test_images/16.jpg'))#8
img = img.to(torch.float)/255
kernel_size_value = 5
gaussian_blur = GaussianBlur(kernel_size_value)
blur_values = [0.5, 0.66, 0.83, 1, 1.33, 1.66, 2, 3, 4, 5]
cv2.imwrite(f'/home/yoni/Desktop/test/b0_orig.png', (img.cpu().numpy()*255).astype(np.uint8))
for idx,sigma in enumerate(blur_values):
  new_img = gaussian_blur(img, sigma)
  cv2.imwrite(f'/home/yoni/Desktop/test/b0_{idx}.png', (new_img.cpu().numpy()*255).astype(np.uint8))
'''
   
def diff_training_sample_ids_btwn_dirs(dir1, dir2):
  diff = 0
  for i in os.listdir(base_dir):
    training_sample_id = i.split('.')[0]
    if not os.path.exists(os.path.join(base_dir2, training_sample_id+'.jpg')):
      diff += 1
  print(diff)
  

base_dir = '/home/yoni/Desktop/f/data/processed_data_vton/same_person_two_poses/schp_raw_output/densepose/'
base_dir2 = '/home/yoni/Desktop/processed_data_vton/same_person_two_poses/person_original/m/'
# diff_training_sample_ids_btwn_dirs(base_dir, base_dir2)    
    

def downsample_mask_arr(arr):
  '''
  Downsample a mask array such that each dimension is half/quarter the size of the original.
  The value of each pixel in the downsampled array (which corresponds to a 2x2 or 4x4 grid in the original array)
  should be 1 if half or more (i.e 2 or 8) of the corresponding original grid are 1s. Otherwise it should be 0.
  '''
  # Reshape the original array to divide it into non-overlapping downsample_factor x downsample_factor blocks
  reshaped_arr = arr[:new_height*downsample_factor, :new_width*downsample_factor].reshape(new_height, downsample_factor, new_width, downsample_factor)
  # Sum along the last two axes to count the number of ones in each downsample_factor x downsample_factor grid
  grid_sum = np.sum(reshaped_arr, axis=(1, 3))
  # Create the downsampled array based on the downsampling logic
  if downsample_factor == 2:
    downsampled_arr = grid_sum >= 2
  else:
    downsampled_arr = grid_sum >= 8
  return downsampled_arr


def create_downsampled_data_from_m(dirs, size='s'):
  
  # Create the smaller dataset by downsampling the "medium" sized data.
  for directory in dirs:
    directory_m = os.path.join(directory, 'm')
    directory_target = os.path.join(directory, size)
    filenames = os.listdir(directory_m)
    file_type = filenames[0].split('.')[-1]
    for filename in tqdm(filenames):
      if file_type == 'jpg':
        file_m = cv2.imread(os.path.join(directory_m, filename))
        file_target = cv2.resize(file_m, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(directory_target, filename), file_target)
      elif file_type == 'npy': # boolean masks
        file_m = np.load(os.path.join(directory_m, filename))
        file_target = downsample_mask_arr(file_m)
        np.save(os.path.join(directory_target, filename), file_target)
      elif file_type == 'txt':
        with open(os.path.join(directory_m, filename), 'r') as f:
          file_m = eval(f.readlines()[0])
          file_target = []
          for coord in file_m:
            if coord is None:
              file_target.append(None)
            else:
              downsampled_coord = (round(coord[0]/downsample_factor), round(coord[1]/downsample_factor))
              file_target.append(downsampled_coord)
          assert len(file_target)==17
        with open(os.path.join(directory_target, filename), 'w') as f:
          f.write(str(file_target))


def create_downsampled_data_from_m_all(size='s'):
  base_dir =  '/home/yoni/Desktop/f/data/processed_data_vton'
  data_sources = ['misc_online', 'multi_pose', 'paired_high_res', 'same_person_two_poses']
  data_sources = ['paired_high_res']
  sub_dirs = ['clothing', 'mask_coordinates', 'person_original', 'person_with_masked_clothing', 'pose_keypoints']
  for data_source in data_sources:
    data_source_path = os.path.join(base_dir, data_source)
    data_source_sub_dirs_paths = []
    for sub_dir in sub_dirs:
      data_source_sub_dirs_paths.append(os.path.join(data_source_path, sub_dir))

    create_downsampled_data_from_m(data_source_sub_dirs_paths, size)
    print(f'finished handling data source: {data_source}')
  
  
def create_augmented_training_sample(person_original_img:np.ndarray, clothing_img:np.ndarray, person_with_masked_clothing_img:np.ndarray, mask_coordinates_arr:np.ndarray, pose_keypoints_list:List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  '''
  Horizontally flip the image and modify the contrast.
  '''
  person_original_img = cv2.flip(person_original_img, 1)
  clothing_img = cv2.flip(clothing_img, 1)
  person_with_masked_clothing_img = cv2.flip(person_with_masked_clothing_img, 1)
  mask_coordinates_arr = cv2.flip(mask_coordinates_arr, 1)

  light_clothing = np.mean(clothing_img[VTON_RESOLUTION[size][0]//2-10:VTON_RESOLUTION[size][0]//2+10, VTON_RESOLUTION[size][1]//2-10:VTON_RESOLUTION[size][1]//2+10]) > 200
  if light_clothing:
    # Random contrast adjustment
    beta = uniform(-30,-50)
    # Random contrast multiplier
    alpha = uniform(0.75, 1.1)
  else:
    beta = uniform(-30,10)
    alpha = uniform(0.8, 1.2)
  
  person_original_img = cv2.convertScaleAbs(person_original_img, alpha=alpha, beta=beta)
  clothing_img = cv2.convertScaleAbs(clothing_img, alpha=alpha, beta=beta)
  person_with_masked_clothing_img = cv2.convertScaleAbs(person_with_masked_clothing_img, alpha=alpha, beta=beta)
  # The masked out pixels should have the same, constant value.
  person_with_masked_clothing_img[mask_coordinates_arr==1] = [128,128,128]
  
  flipped_pose_keypoints_list = []
  img_width = person_original_img.shape[1] - 1
  for coord in pose_keypoints_list:
    if coord is None:
      flipped_pose_keypoints_list.append(None)
    else:
      # Note that the order of the pose coordinates is reversed from numpy, since it was optimized for cv2.
      flipped_coord = (img_width - coord[0], coord[1])
      flipped_pose_keypoints_list.append(flipped_coord)

  return (person_original_img, clothing_img, person_with_masked_clothing_img, flipped_pose_keypoints_list)


COLOR_AQUA = (255, 255, 0) # BGR format
def create_final_dataset_vton_size_to_size(size='s'):
  ready_datasets = '/home/yoni/Desktop/f/data/ready_datasets'
  target_dir = os.path.join(ready_datasets, f'vton_{size}_to_{size}')
  os.makedirs(target_dir, exist_ok=True)
  target_inspection_dir = os.path.join(ready_datasets, f'vton_{size}_to_{size}_inspection')
  os.makedirs(target_inspection_dir, exist_ok=True)
  log_filepath = os.path.join(ready_datasets, f'vton_{size}_to_{size}_log.txt')
  data_sources = ['misc_online', 'multi_pose', 'paired_high_res', 'same_person_two_poses']
  data_sources = ['paired_high_res']
  # How many additional (augmented) training samples should be created 
  # from an original training sample, coming from a particular data source.
  # Integer values {1,2,3,...}, mean # requested samples.
  # Fractional values [0, 1], are the probability of creating a single sample.
  # prob_aug = {'misc_online': 1, 'multi_pose': 0.5, 'paired_high_res':0.5, 'same_person_two_poses':1}
  prob_aug = {'misc_online': 1, 'multi_pose': 0.5, 'paired_high_res':1, 'same_person_two_poses':1}
  num_training_samples = 0
  
  with open(log_filepath, 'w') as log_file:
    for data_source_dir_name in data_sources:
      num_training_samples_before_this_data_source = num_training_samples
      person_original_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'person_original', size)
      clothing_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'clothing', size)
      pose_keypoints_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'pose_keypoints', size)
      person_with_masked_clothing_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'person_with_masked_clothing', size)
      mask_coordinates_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'mask_coordinates', size)
      
      def _normalize_and_save_training_sample(person_original_img:np.ndarray, clothing_img:np.ndarray, person_with_masked_clothing_img:np.ndarray, mask_coordinates_arr:np.ndarray, pose_keypoints_list:List, training_sample_id_final:str, inspect:bool):
        person_original_filepath_final = os.path.join(target_dir, training_sample_id_final + '_person.pth')
        clothing_filepath_final = os.path.join(target_dir, training_sample_id_final + '_clothing.pth')
        person_with_masked_clothing_filepath_final = os.path.join(target_dir, training_sample_id_final + '_masked.pth')
        mask_coordinates_filepath_final = os.path.join(target_dir, training_sample_id_final + '_mask-coords.pth')
        pose_keypoints_filepath_final = os.path.join(target_dir, training_sample_id_final + '_pose.txt')
        # Normalize to [-1,1].
        person_original_img_norm = (person_original_img / 127.5) - 1
        clothing_img_norm = (clothing_img / 127.5) - 1
        person_with_masked_clothing_img_norm = (person_with_masked_clothing_img / 127.5) - 1
        # Permute order of dimensions so that the channel is first.
        person_original_img_norm = person_original_img_norm.transpose(2,0,1)
        clothing_img_norm = clothing_img_norm.transpose(2,0,1)
        person_with_masked_clothing_img_norm = person_with_masked_clothing_img_norm.transpose(2,0,1)
        # Cast to float 16, and reverse the order of dimensions from BGR to RGB.
        person_original_img_norm = np.copy(person_original_img_norm.astype(np.float16)[::-1])
        clothing_img_norm = np.copy(clothing_img_norm.astype(np.float16)[::-1])
        person_with_masked_clothing_img_norm = np.copy(person_with_masked_clothing_img_norm.astype(np.float16)[::-1])
        torch.save(torch.tensor(person_original_img_norm, dtype=torch.bfloat16), person_original_filepath_final)
        torch.save(torch.tensor(clothing_img_norm, dtype=torch.bfloat16), clothing_filepath_final)
        torch.save(torch.tensor(person_with_masked_clothing_img_norm, dtype=torch.bfloat16), person_with_masked_clothing_filepath_final)
        torch.save(torch.tensor(mask_coordinates_arr).bool(), mask_coordinates_filepath_final)
        
        with open(pose_keypoints_filepath_final, 'w') as pose_keypoints_file:
          pose_keypoints_file.write(str(pose_keypoints_list))
        if inspect:
          person_original_filepath_final = os.path.join(target_inspection_dir, training_sample_id_final + '_person.jpg')
          clothing_filepath_final = os.path.join(target_inspection_dir, training_sample_id_final + '_clothing.jpg')
          person_with_masked_clothing_filepath_final = os.path.join(target_inspection_dir, training_sample_id_final + '_masked.jpg')
          person_with_keypoints_filepath_final = os.path.join(target_inspection_dir, training_sample_id_final + '_pose.jpg')
          cv2.imwrite(person_original_filepath_final, person_original_img)
          cv2.imwrite(clothing_filepath_final, clothing_img)
          cv2.imwrite(person_with_masked_clothing_filepath_final, person_with_masked_clothing_img)
          person_for_keypoints_img = np.copy(person_original_img)
          for coord in pose_keypoints_list:
            if coord is not None:
                cv2.circle(person_for_keypoints_img, coord, radius=2, color=COLOR_AQUA, thickness=-1)
            cv2.imwrite(person_with_keypoints_filepath_final, person_for_keypoints_img)
      
      for person_original_filename in tqdm(os.listdir(person_original_dir)):
        training_sample_id_original = person_original_filename.split('.')[0]
        person_original_filepath = os.path.join(person_original_dir, training_sample_id_original + '.jpg')
        clothing_filepath = os.path.join(clothing_dir, training_sample_id_original + '.jpg')
        pose_keypoints_filepath = os.path.join(pose_keypoints_dir, training_sample_id_original + '.txt')
        person_with_masked_clothing_filepath = os.path.join(person_with_masked_clothing_dir, training_sample_id_original + '.jpg')
        mask_coordinates_filepath = os.path.join(mask_coordinates_dir, training_sample_id_original + '.npy')
        
        all_data_exists_for_training_sample_id = True
        for filepath in [person_original_filepath, clothing_filepath, pose_keypoints_filepath, person_with_masked_clothing_filepath, mask_coordinates_filepath]:
          if not os.path.exists(filepath):
            log_file.write(f'incomplete sample {training_sample_id_original}\n')
            all_data_exists_for_training_sample_id = False
            break
        if not all_data_exists_for_training_sample_id:
          continue
        
        inspect = random() < (prob_aug[data_source_dir_name] * 0.1 + 0.002)
        
        person_original_img = cv2.imread(person_original_filepath)
        clothing_img = cv2.imread(clothing_filepath)
        person_with_masked_clothing_img = cv2.imread(person_with_masked_clothing_filepath)
        with open(pose_keypoints_filepath, 'r') as pose_keypoints_file:
          pose_keypoints_list = eval(pose_keypoints_file.readlines()[0])
        training_sample_id_final = training_sample_id_original + f'_{num_training_samples}_orig'
        mask_coordinates_arr = np.load(mask_coordinates_filepath).astype(np.uint8)
        _normalize_and_save_training_sample(person_original_img, clothing_img, person_with_masked_clothing_img, mask_coordinates_arr, pose_keypoints_list, training_sample_id_final, inspect)
        num_training_samples += 1
                
        num_augmentations = prob_aug[data_source_dir_name]
        if num_augmentations >= 1:
          for _ in range(num_augmentations):
            augmented_sample = create_augmented_training_sample(person_original_img, clothing_img, person_with_masked_clothing_img, mask_coordinates_arr, pose_keypoints_list)
            person_original_img_aug, clothing_img_aug, person_with_masked_clothing_img_aug, pose_keypoints_list_aug = augmented_sample
            training_sample_id_final = training_sample_id_original + f'_{num_training_samples}_aug'
            _normalize_and_save_training_sample(person_original_img_aug, clothing_img_aug, person_with_masked_clothing_img_aug, mask_coordinates_arr, pose_keypoints_list_aug, training_sample_id_final, inspect)
            num_training_samples += 1
        elif num_augmentations >= 0:
          if random() < num_augmentations:
            augmented_sample = create_augmented_training_sample(person_original_img, clothing_img, person_with_masked_clothing_img, mask_coordinates_arr, pose_keypoints_list)
            person_original_img_aug, clothing_img_aug, person_with_masked_clothing_img_aug, pose_keypoints_list_aug = augmented_sample
            training_sample_id_final = training_sample_id_original + f'_{num_training_samples}_aug'
            _normalize_and_save_training_sample(person_original_img_aug, clothing_img_aug, person_with_masked_clothing_img_aug, mask_coordinates_arr, pose_keypoints_list_aug, training_sample_id_final, inspect)
            num_training_samples += 1
      
        # if num_training_samples > 100:
        #   return
      
      print(f'finished {data_source_dir_name}, processed {num_training_samples - num_training_samples_before_this_data_source} samples') 
  print(f'FINISHED, total of {num_training_samples} samples')      


downsample_factor_per_size = {'s':2, 't':4}
for size in ['t']:
  new_height = VTON_RESOLUTION[size][0]
  new_width = VTON_RESOLUTION[size][1]
  downsample_factor = downsample_factor_per_size[size]

  # create_downsampled_data_from_m_all(size=size)
  create_final_dataset_vton_size_to_size(size=size)
