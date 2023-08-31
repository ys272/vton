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
import torch
import torch.nn as nn
import torch.nn.functional as F

base_dir_src = ''
base_dir_dst = ''

counter = 0
for d in os.listdir(base_dir_src):
  filenames = set(os.listdir(os.path.join(base_dir_src,d)))
  final_filenames = set()
  for filename in filenames:
    sample_id = '_'.join(filename.split('_')[:-1])
    suffix = filename.split('_')[-1]
    person = sample_id + '_person'
    clothing = sample_id + '_clothing'
    if suffix  == 'person' and clothing in filenames:
        final_filenames.add(sample_id)
    elif suffix  == 'clothing' and person in filenames:
        final_filenames.add(sample_id)
  
  for sample_id in final_filenames:
    clothing = cv2.imread(os.path.join(base_dir_src, d, sample_id+'_clothing.jpg'))
    person = cv2.imread(os.path.join(base_dir_src, d, sample_id+'_person.jpg'))
    cv2.imwrite(os.path.join(base_dir_dst, 'clothing', 'm', sample_id + f'{counter}.jpg'), person)
    cv2.imwrite(os.path.join(base_dir_dst, 'person', 'm', sample_id + f'{counter}.jpg'), clothing)
    counter += 1
    
    

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
