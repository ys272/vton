import os
import numpy as np
import cv2
from tqdm import tqdm
import sys
from utils import resize_img
import pickle
import config as c
from pose import PoseModel
from schp import extract_person_without_clothing
import torch

# base_dir = '/home/yoni/Desktop/f/data/processed_data_vton/same_person_two_poses/schp_raw_output/densepose/'
# base_dir2 = '/home/yoni/Desktop/processed_data_vton/same_person_two_poses/person_original/m/'
# diff = 0
# for i in os.listdir(base_dir):
#   ii = i.split('.')[0]
#   if not os.path.exists(os.path.join(base_dir2, ii+'.jpg')):
#     diff += 1
#     # print(ii)
# print(diff)

# base_dir = '/home/yoni/Desktop/processed_data_vton/multi_pose/person_original/m3'
# listoffiles = os.listdir(base_dir)
# for filename in listoffiles[:len(listoffiles)//3]:
#   os.remove(os.path.join(base_dir, filename))

# for filename in listoffiles[2*len(listoffiles)//3:]:
#   os.remove(os.path.join(base_dir, filename))
base_dir='/home/yoni/Desktop/f/data/processed_data_vton/multi_pose/schp_raw_output/densepose'
with open(base_dir+'/densepose3.pkl', 'rb') as f:
    results = torch.load(f)
for i in range(len(results)):
    filepath = results[i]['file_name']
    training_sample_id = filepath.split('/')[-1].split('.')[0]
    file_name = os.path.join(base_dir, f'{training_sample_id}.pkl')
    with open(file_name, 'wb') as file:
      torch.save(results[i], file)

