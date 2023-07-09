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



# base_dir = '/home/yoni/Desktop/processed_data_vton/multi_pose/person_original/m3'
# listoffiles = os.listdir(base_dir)
# for filename in listoffiles[:len(listoffiles)//3]:
#   os.remove(os.path.join(base_dir, filename))

# for filename in listoffiles[2*len(listoffiles)//3:]:
#   os.remove(os.path.join(base_dir, filename))


# base_dir='/home/yoni/Desktop/f/data/processed_data_vton/multi_pose/schp_raw_output/densepose'
# with open(base_dir+'/densepose3.pkl', 'rb') as f:
#     results = torch.load(f)
# for i in range(len(results)):
#     filepath = results[i]['file_name']
#     training_sample_id = filepath.split('/')[-1].split('.')[0]
#     file_name = os.path.join(base_dir, f'{training_sample_id}.pkl')
#     with open(file_name, 'wb') as file:
#       torch.save(results[i], file)


# def remove_duplicates():
#     sizes = []
#     filenames = []
#     for filename in tqdm(os.listdir('/home/yoni/Desktop/m/')):
#         path = os.path.join('/home/yoni/Desktop/m/', filename)
#         filenames.append(filename)
#         sizes.append(os.path.getsize(path))

#     # Sort the lists by size.
#     sorted_lists = sorted(zip(sizes, filenames))
#     sizes, filenames = zip(*sorted_lists)

#     # The first index at which each unique size appears.
#     _, first_indices_where_unique_size_appears = np.unique(sizes, return_index=True)
#     duplicates_to_remove = set()
#     for starting_index_for_this_unique_size in tqdm(range(len(first_indices_where_unique_size_appears[:-1]))):
#         # Because the size array (in which we search for unique integers) was sorted, we know that the files in the range
#         # below all have the same (compressed) byte size, so we should compare them all to each other for equality.
#         for i in range(first_indices_where_unique_size_appears[starting_index_for_this_unique_size], first_indices_where_unique_size_appears[starting_index_for_this_unique_size+1]):
#             first_image = cv2.imread(os.path.join('/home/yoni/Desktop/m/', filenames[i]))
#             print(os.path.join('/home/yoni/Desktop/m/', filenames[i]))
#             # Compare `first_image` to all the images that follow it (that have the same size).
#             for j in range(i+1, first_indices_where_unique_size_appears[starting_index_for_this_unique_size+1]):
#                 second_image = cv2.imread(os.path.join('/home/yoni/Desktop/m/', filenames[j]))
#                 if np.all(first_image == second_image):
#                     # If identical, remove the first one.
#                     duplicates_to_remove.add(filenames[i].split('.')[0])
#                     # No need to search for any further matches for the first image, since it's already been removed.
#                     break
    
#     print(f'!!! Removing {len(duplicates_to_remove)} training samples !!!')
# remove_duplicates()
