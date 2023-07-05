import os
import numpy as np
import cv2
from tqdm import tqdm
import sys
from utils import resize_img

with open('/home/yoni/Desktop/f/data/processed_data_vton/multi_pose/log_person_detection_29.txt', 'r') as f:
  for line in f.readlines():
    a = line.split(' ')[-1].strip()
    d = '/home/yoni/Desktop/f/data/processed_data_vton/multi_pose/person_original/m'  
    if os.path.exists(os.path.join(d,a)+'.jpg'):
      os.remove(os.path.join(d,a)+'.jpg')
    d = '/home/yoni/Desktop/f/data/processed_data_vton/multi_pose/person_original/l'  
    if os.path.exists(os.path.join(d,a)+'.jpg'):os.remove(os.path.join(d,a)+'.jpg')
    d = '/home/yoni/Desktop/f/data/processed_data_vton/multi_pose/person_with_masked_clothing/m'  
    if os.path.exists(os.path.join(d,a)+'.jpg'):os.remove(os.path.join(d,a)+'.jpg')
    d = '/home/yoni/Desktop/f/data/processed_data_vton/multi_pose/clothing/m'
    if os.path.exists(os.path.join(d,a)+'.jpg'):os.remove(os.path.join(d,a)+'.jpg')
    d = '/home/yoni/Desktop/f/data/processed_data_vton/multi_pose/mask_coordinates/m'  
    if os.path.exists(os.path.join(d,a)+'.txt'):os.remove(os.path.join(d,a)+'.txt')
    d = '/home/yoni/Desktop/f/data/processed_data_vton/multi_pose/pose_keypoints/m'  
    if os.path.exists(os.path.join(d,a)+'.txt'):os.remove(os.path.join(d,a)+'.txt')
