import os
import numpy as np
import cv2
from tqdm import tqdm
import sys
from utils import resize_img

a='/home/yoni/Desktop/f/data/processed_data_vton/multi_pose/inspection/multipose_id_00000911_52_person_original.jpg'
from pose import PoseModel
p = PoseModel()
a = cv2.imread(a)
# b=resize_img(256,256,img=a)
# cv2.imwrite('/home/yoni/Desktop/aaaaa.jpg',b)
print('---',a.shape)
k=p.get_keypoints(a)
o = '/home/yoni/Desktop/aa.png'
p.save_or_return_img_w_overlaid_keypoints(a, k, o)

# # a = cv2.imread('/home/yoni/Desktop/multipose_id_00004372_27037.jpg')
# a = cv2.imread('/home/yoni/Desktop/3af3a8c1e0581011f46326d3b837a53d.jpg')
# aa = resize_image_with_padding(800, 1200, img=a)
# cv2.imwrite('/home/yoni/Desktop/03_1_front_-.jpg', aa)
# sys.exit()
# # with open('/home/yoni/Desktop/f/data/processed_data_vton/same_person_two_poses/pose_keypoints/sp2p_299_86.txt', 'r') as f:
# #     # f.readlines()
# #     breakpoint()
#     # print('d')
# # 

# a='/home/yoni/Desktop/f/data/processed_data_vton/same_person_two_poses/schp_raw_output/sp2p_299_86.npy'

# d = '/home/yoni/Desktop/f/data/processed_data_vton/same_person_two_poses2/person_original/'
# sizes = []
# filenames = []
# for p in tqdm(os.listdir(d)):
#     path = d+p
#     filenames.append(p)
#     # n = np.load(path, allow_pickle=True)
#     # n = cv2.imread(path)
#     sizes.append(os.path.getsize(path))

# sorted_lists = sorted(zip(sizes,filenames))
# s1,s2 = zip(*sorted_lists)

# old_size = -1
# equal = 0
# to_delete = []
# for i,size in tqdm(enumerate(s1[:-1])):
#     if s1[i] == s1[i+1]:
#         first=cv2.imread(d+s2[i])
#         second=cv2.imread(d+s2[i+1])
#         if np.all(first==second):
#             to_delete.append(first)
            

# print('f')
# print('f')
    

