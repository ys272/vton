import shutil
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
import torch
import torch.nn as nn
import torch.nn.functional as F


# with open('/home/yoni/Desktop/f/data/processed_data_vton/artistic/schp_raw_output/densepose/densepose.pkl', 'rb') as f:
#     data = torch.load(f)
#     for densefile in data:
#       filename = densefile['file_name'].split('/')[-1].split('.')[0]
#       torch.save(densefile, f'/home/yoni/Desktop/f/data/processed_data_vton/artistic/schp_raw_output/densepose/{filename}.pkl')
# sys.exit()


# base_dir_src = '/home/yoni/Desktop/artistic/'
# base_dir_dst = '/home/yoni/Desktop/f/data/original_data/artistic/'

# os.makedirs(os.path.join(base_dir_dst, 'clothing'), exist_ok=True)
# os.makedirs(os.path.join(base_dir_dst, 'person'), exist_ok=True)
# counter = 0
# for d in tqdm(os.listdir(base_dir_src)):
#   filenames = set(os.listdir(os.path.join(base_dir_src,d)))
#   final_filenames = set()
#   for filename in filenames:
#     sample_id = '_'.join(filename.split('_')[:-1])
#     suffix = filename.split('_')[-1].split('.')[0]
#     person = sample_id + '_person.jpg'
#     clothing = sample_id + '_clothing.jpg'
#     if suffix  == 'person' and clothing in filenames:
#         final_filenames.add(sample_id)
#     elif suffix  == 'clothing' and person in filenames:
#         final_filenames.add(sample_id)
  
#   for sample_id in final_filenames:
#     clothing_path = os.path.join(base_dir_src, d, sample_id+'_clothing.jpg')
#     person_path = os.path.join(base_dir_src, d, sample_id+'_person.jpg')
#     if os.path.getsize(clothing_path) < 1000 or os.path.getsize(person_path) < 1000:
#       continue
#     clothing = cv2.imread(clothing_path)
#     person = cv2.imread(person_path)
#     cv2.imwrite(os.path.join(base_dir_dst, 'clothing', f'{counter}.jpg'), clothing)
#     cv2.imwrite(os.path.join(base_dir_dst, 'person', f'{counter}.jpg'), person)
#     counter += 1
    

# basedir = '/home/yoni/Desktop/f/data/processed_data_vton/same_person_two_poses/'
# destdir = '/home/yoni/Desktop/f/data/filtering/sp2p'

# for i,filename in enumerate(os.listdir('/home/yoni/Desktop/f/data/processed_data_vton/same_person_two_poses/clothing/m')):
#   dir_idx = str(int(i / 1000))
#   dir_name = os.path.join(destdir, dir_idx)
#   os.makedirs(dir_name, exist_ok=True)
#   clothing = cv2.imread(os.path.join(basedir, 'clothing/m', filename))
#   mask = cv2.imread(os.path.join(basedir, 'person_with_masked_clothing/m', filename))
#   person = cv2.imread(os.path.join(basedir, 'person_original/m', filename))
#   filename = filename.split('.')[0]
#   height, width, _ = clothing.shape
#   combined_image = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)
#   combined_image[:height, :width] = clothing
#   combined_image[:height, width:] = mask
#   combined_image[height:, :width] = person
#   cv2.imwrite(os.path.join(dir_name, filename+'.jpg'), combined_image)
  # cv2.imwrite(os.path.join(dir_name, filename+'_c.jpg'), clothing)
  # cv2.imwrite(os.path.join(dir_name, filename+'_m.jpg'), mask)
  # cv2.imwrite(os.path.join(dir_name, filename+'_p.jpg'), person)
  # if i >=12:
  #   break


# raw_data_dir = '/home/yoni/Desktop/f/data/processed_data_vton/same_person_two_poses/'
# quarantined_data_dir = '/home/yoni/Desktop/f/data/processed_data_vton/same_person_two_poses/quarantined'
# filtered_data_dir = '/home/yoni/Desktop/f/data/filtering/sp2p DONE'
# subdirs = ['clothing', 'mask_coordinates', 'person_original', 'person_with_masked_clothing', 'pose_keypoints']

# s = set()
# for directory in os.listdir(filtered_data_dir):
#   for filename in os.listdir(os.path.join(filtered_data_dir, directory)):
#     sample_id = filename.split('.')[0]
#     s.add(sample_id)

# for subdir in subdirs:
#   counter=0
#   for filename in os.listdir(os.path.join(raw_data_dir, subdir, 'm')):
#     file_id = filename.split('.')[0]
#     if file_id not in s:
#       counter += 1
#       new_dir = os.path.join(quarantined_data_dir, subdir, 'm')
#       os.makedirs(new_dir,exist_ok=True)
#       shutil.copy(os.path.join(raw_data_dir, subdir, 'm', filename), os.path.join(new_dir, filename))
#       os.remove(os.path.join(raw_data_dir, subdir, 'm', filename))
      
#   print(subdir, counter)
