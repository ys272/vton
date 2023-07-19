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


def improve_contrast_if_very_light(clothing_img, person_original_img):
    is_very_light_clothing = np.mean(clothing_img[clothing_img.shape[0]//2-10:clothing_img.shape[0]//2+10, clothing_img.shape[1]//2-10:clothing_img.shape[1]//2+10]) > 200
    if is_very_light_clothing:
        # Random contrast adjustment
        beta = -5#uniform(-10,-20)
        # Random contrast multiplier
        alpha = 1#uniform(0.75, 1.1)
        
        clothing_img = cv2.convertScaleAbs(clothing_img, alpha=alpha, beta=beta)
        person_original_img = cv2.convertScaleAbs(person_original_img, alpha=alpha, beta=beta)
    return clothing_img, person_original_img
    

def improve_contrast_process(clothing_dir, person_original_dir):
    for filename in tqdm(os.listdir(clothing_dir)):
        clothing_img_path = os.path.join(clothing_dir, filename)
        person_original_img_path = os.path.join(person_original_dir, filename)
        clothing_img = cv2.imread(clothing_img_path)
        person_original_img = cv2.imread(person_original_img_path)
        clothing_img, person_original_img = improve_contrast_if_very_light(clothing_img, person_original_img)
        cv2.imwrite(clothing_img_path, clothing_img)
        cv2.imwrite(person_original_img_path, person_original_img)
        
        
new_height = VTON_RESOLUTION['s'][0]
new_width = VTON_RESOLUTION['s'][1]
def downsample_mask_arr(arr):
  '''
  Downsample a mask array such that each dimension is half the size of the original.
  The value of each pixel in the downsampled array (which corresponds to a 2x2 grid in the original array)
  should be 1 if there are two or more 1s in the original array. Otherwise it should be 0.
  '''
  # Reshape the original array to divide it into non-overlapping 2x2 blocks
  reshaped_arr = arr[:new_height*2, :new_width*2].reshape(new_height, 2, new_width, 2)
  # Sum along the last two axes to count the number of ones in each 2x2 grid
  grid_sum = np.sum(reshaped_arr, axis=(1, 3))
  # Create the downsampled array based on the downsampling logic
  downsampled_arr = grid_sum >= 2
  return downsampled_arr


def create_downsampled_s_data_from_m(dirs):
  # Create the "small" dataset by downsampling the "medium" sized data.
  for directory in dirs:
    directory_m = os.path.join(directory, 'm')
    directory_s = os.path.join(directory, 's')
    filenames = os.listdir(directory_m)
    file_type = filenames[0].split('.')[-1]
    for filename in tqdm(filenames):
      if file_type == 'jpg':
        file_m = cv2.imread(os.path.join(directory_m, filename))
        file_s = cv2.resize(file_m, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(directory_s, filename), file_s)
      elif file_type == 'npy': # boolean masks
        file_m = np.load(os.path.join(directory_m, filename))
        file_s = downsample_mask_arr(file_m)
        np.save(os.path.join(directory_s, filename), file_s)
      elif file_type == 'txt':
        with open(os.path.join(directory_m, filename), 'r') as f:
          file_m = eval(f.readlines()[0])
          file_s = []
          for coord in file_m:
            if coord is None:
              file_s.append(None)
            else:
              downsampled_coord = (round(coord[0]/2), round(coord[1]/2))
              file_s.append(downsampled_coord)
          assert len(file_s)==17
        with open(os.path.join(directory_s, filename), 'w') as f:
          f.write(str(file_s))


def create_downsampled_s_data_from_m_all():
  base_dir =  '/home/yoni/Desktop/f/data/processed_data_vton'
  data_sources = ['misc_online', 'multi_pose', 'paired_high_res', 'same_person_two_poses']
  sub_dirs = ['clothing', 'mask_coordinates', 'person_original', 'person_with_masked_clothing', 'pose_keypoints']
  for data_source in data_sources:
    data_source_path = os.path.join(base_dir, data_source)
    data_source_sub_dirs_paths = []
    for sub_dir in sub_dirs:
      data_source_sub_dirs_paths.append(os.path.join(data_source_path, sub_dir))

    create_downsampled_s_data_from_m(data_source_sub_dirs_paths)
    print(f'finished handling data source: {data_source}')

# create_downsampled_s_data_from_m_all()
  
  
def create_augmented_training_sample(person_original_img:np.ndarray, clothing_img:np.ndarray, person_with_masked_clothing_img:np.ndarray, mask_coordinates_arr:np.ndarray, pose_keypoints_list:List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  '''
  Horizontally flip the image and modify the contrast.
  '''
  person_original_img = cv2.flip(person_original_img, 1)
  clothing_img = cv2.flip(clothing_img, 1)
  person_with_masked_clothing_img = cv2.flip(person_with_masked_clothing_img, 1)
  mask_coordinates_arr = cv2.flip(mask_coordinates_arr, 1)

  light_clothing = np.mean(clothing_img[VTON_RESOLUTION['s'][0]//2-10:VTON_RESOLUTION['s'][0]//2+10, VTON_RESOLUTION['s'][1]//2-10:VTON_RESOLUTION['s'][1]//2+10]) > 200
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
def create_final_dataset_vton_s_to_s():
  ready_datasets = '/home/yoni/Desktop/f/data/ready_datasets'
  target_dir = os.path.join(ready_datasets, 'vton_s_to_s')
  os.makedirs(target_dir, exist_ok=True)
  target_inspection_dir = os.path.join(ready_datasets, 'vton_s_to_s_inspection')
  os.makedirs(target_inspection_dir, exist_ok=True)
  log_filepath = os.path.join(ready_datasets, 'vton_s_to_s_log.txt')
  data_sources = ['misc_online', 'multi_pose', 'paired_high_res', 'same_person_two_poses']
  # How many additional (augmented) training samples should be created 
  # from an original training sample, coming from a particular data source.
  # Integer values {1,2,3,...}, mean # requested samples.
  # Fractional values [0, 1], are the probability of creating a single sample.
  prob_aug = {'misc_online': 1, 'multi_pose': 0.1, 'paired_high_res':0, 'same_person_two_poses':0.15}
  num_training_samples = 0
  
  with open(log_filepath, 'w') as log_file:
    for data_source_dir_name in data_sources:
      num_training_samples_before_this_data_source = num_training_samples
      person_original_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'person_original', 's')
      clothing_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'clothing', 's')
      pose_keypoints_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'pose_keypoints', 's')
      person_with_masked_clothing_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'person_with_masked_clothing', 's')
      mask_coordinates_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'mask_coordinates', 's')
      
      def _save_training_sample(person_original_img:np.ndarray, clothing_img:np.ndarray, person_with_masked_clothing_img:np.ndarray, pose_keypoints_list:List, training_sample_id_final:str, inspect:bool):
        person_original_filepath_final = os.path.join(target_dir, training_sample_id_final + '_person_original.npy')
        clothing_filepath_final = os.path.join(target_dir, training_sample_id_final + '_clothing.npy')
        person_with_masked_clothing_filepath_final = os.path.join(target_dir, training_sample_id_final + '_person_with_masked_clothing.npy')
        pose_keypoints_filepath_final = os.path.join(target_dir, training_sample_id_final + '_pose.txt')
        np.save(person_original_filepath_final, person_original_img)
        np.save(clothing_filepath_final, clothing_img)
        np.save(person_with_masked_clothing_filepath_final, person_with_masked_clothing_img)
        with open(pose_keypoints_filepath_final, 'w') as pose_keypoints_file:
          pose_keypoints_file.write(str(pose_keypoints_list))
        if inspect:
          person_original_filepath_final = os.path.join(target_inspection_dir, training_sample_id_final + '_person_original.jpg')
          clothing_filepath_final = os.path.join(target_inspection_dir, training_sample_id_final + '_clothing.jpg')
          person_with_masked_clothing_filepath_final = os.path.join(target_inspection_dir, training_sample_id_final + '_person_with_masked_clothing.jpg')
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
        _save_training_sample(person_original_img, clothing_img, person_with_masked_clothing_img, pose_keypoints_list, training_sample_id_final, inspect)
        num_training_samples += 1
                
        num_augmentations = prob_aug[data_source_dir_name]
        if num_augmentations >= 1:
          mask_coordinates_arr = np.load(mask_coordinates_filepath).astype(np.uint8)
          for _ in range(num_augmentations):
            augmented_sample = create_augmented_training_sample(person_original_img, clothing_img, person_with_masked_clothing_img, mask_coordinates_arr, pose_keypoints_list)
            person_original_img_aug, clothing_img_aug, person_with_masked_clothing_img_aug, pose_keypoints_list_aug = augmented_sample
            training_sample_id_final = training_sample_id_original + f'_{num_training_samples}_aug'
            _save_training_sample(person_original_img_aug, clothing_img_aug, person_with_masked_clothing_img_aug, pose_keypoints_list_aug, training_sample_id_final, inspect)
            num_training_samples += 1
        elif num_augmentations >= 0:
          mask_coordinates_arr = np.load(mask_coordinates_filepath).astype(np.uint8)
          if random() < num_augmentations:
            augmented_sample = create_augmented_training_sample(person_original_img, clothing_img, person_with_masked_clothing_img, mask_coordinates_arr, pose_keypoints_list)
            person_original_img_aug, clothing_img_aug, person_with_masked_clothing_img_aug, pose_keypoints_list_aug = augmented_sample
            training_sample_id_final = training_sample_id_original + f'_{num_training_samples}_aug'
            _save_training_sample(person_original_img_aug, clothing_img_aug, person_with_masked_clothing_img_aug, pose_keypoints_list_aug, training_sample_id_final, inspect)
            num_training_samples += 1
      
        # if num_training_samples > 100:
        #   return
      
      print(f'finished {data_source_dir_name}, processed {num_training_samples - num_training_samples_before_this_data_source} samples') 
  print(f'FINISHED, total of {num_training_samples} samples')      


create_final_dataset_vton_s_to_s()
