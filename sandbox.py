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
# base_dir = '/home/yoni/Desktop/f/data/processed_data_vton/paired_high_res/schp_raw_output/densepose/'
# with open(os.path.join(base_dir, 'densepose1.pkl'), 'rb') as f:
#   results=torch.load(f)

# for i in range(len(results)):
#     filepath = results[i]['file_name']
#     training_sample_id = filepath.split('/')[-1].split('.')[0]
#     file_name = os.path.join(base_dir, f'{training_sample_id}.pkl')
#     with open(file_name, 'wb') as file:
#       torch.save(results[i], file)


# for filename in os.listdir('/home/yoni/Desktop/input/'):
#   filename=filename.split('.')[0]
#   base_img = cv2.imread('/home/yoni/Desktop/input/'+filename+'.jpg')
#   atr = np.load('/home/yoni/Desktop/atroutput/'+filename+'.npy', allow_pickle=True)
#   atr = np.argmax(atr, axis=-1)
#   pascal = np.load('/home/yoni/Desktop/pascaloutput/'+filename+'.npy', allow_pickle=True)
#   pascal = np.argmax(pascal, axis=-1)
#   atr_hands=[]
#   atr_hair=2
#   atr_skin=11
#   pascal_head=1
#   kernel = np.ones((3, 3), dtype=np.uint8)
#   clothing_mask = (atr==atr_hair ) | ((atr==atr_skin) &(pascal == pascal_head))
#   clothing_mask = clothing_mask.astype(np.uint8)
#   clothing_mask = cv2.dilate(clothing_mask, kernel, iterations=5)
#   clothing_mask = cv2.erode(clothing_mask, kernel, iterations=5)
#   # base_img[clothing_mask != 1] = [255,255,255]
#   # cv2.imwrite('/home/yoni/Desktop/output_/'+filename+'.png', base_img)
#   with open('/home/yoni/Desktop/denseoutput/output_seg.pkl', 'rb') as file:d = torch.load(file)
#   img,_=extract_person_without_clothing(atr, img=np.copy(base_img))
#   img[clothing_mask==1] = base_img[clothing_mask==1]
  
#   cv2.imwrite('/home/yoni/Desktop/output_fix/'+filename+'.png', img)