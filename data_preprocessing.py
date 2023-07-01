import numpy as np
import os
import cv2
import config as c
from utils import resize_image_with_padding
from pose import is_person_detected, get_keypoints, save_img_w_overlaid_keypoints
from schp import run_schp, extract_person_without_clothing
from random import random


def preprocess_fto_pose():
    data_source_acronym = 'fto'
    data_source_dir_name = 'fto'
    person_original_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'person_original')
    clothing_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'clothing')
    pose_keypoints_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'pose_keypoints')
    inspection_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'inspection')
    original_data_dir = os.path.join(c.ORIGINAL_DATA_DIR, data_source_dir_name)
    log_file_path = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'log.txt')

    training_sample_num = 0
    with open(log_file_path, 'a') as log_file:
      for sub_dir_name in ['train', 'test']:
        original_data_sub_dir = os.path.join(original_data_dir, sub_dir_name)
        numbered_dir_names = os.listdir(original_data_sub_dir)
        for dir_number in numbered_dir_names:
            # print(sub_dir_name, dir_number)
            training_sample_dir = os.path.join(original_data_sub_dir, dir_number)
            training_sample_dir_contents = os.listdir(training_sample_dir)
            if len(training_sample_dir_contents) != 6:
                log_file.write(f'6, {dir_number}, {training_sample_dir_contents}\n')
                continue
            '''
            Each folder should contain 2 directories (which we don't need) and 4 images.
            One of the images is the mask, which we don't need.
            One of the images is the standalone clothing.
            The two other images are images of the same person in different poses.
            They can each be paired with the clothing, to provide two independent training samples.
            
            First however, verify that the person detector detects a person in the 
            two image files and fails in the clothing file.
            '''
            person_img_paths = [img_path for img_path in training_sample_dir_contents if img_path not in set(['mask.jpg', 'target.jpg']) and len(img_path.split('.')) > 1]
            if len(person_img_paths) != 2:
                log_file.write(f'2, {dir_number}, {training_sample_dir_contents}\n')
                continue
            person_imgs_and_keypoints = []
            for person_img_path in person_img_paths:
                person_img_path = os.path.join(training_sample_dir, person_img_path)
                person_img = resize_image_with_padding(c.VTON_RESOLUTION, c.VTON_RESOLUTION, input_img_path=person_img_path)
                keypoints = get_keypoints(person_img)
                if not keypoints:
                  log_file.write(f'k, {dir_number}\n')
                  break
                person_imgs_and_keypoints.append((person_img, keypoints))
            if len(person_imgs_and_keypoints) != 2:
                continue
            clothing_img_path = os.path.join(training_sample_dir, 'target.jpg')
            clothing_img = resize_image_with_padding(c.VTON_RESOLUTION, c.VTON_RESOLUTION, input_img_path=clothing_img_path)
            save_for_inspection = True #random() < 0.01
            for (img,keypoints) in person_imgs_and_keypoints:
                training_sample_id = f'{data_source_acronym}_{dir_number}_{training_sample_num}'
                person_original_img_save_path = os.path.join(person_original_dir, f'{training_sample_id}.jpg')
                clothing_img_save_path = os.path.join(clothing_dir, f'{training_sample_id}.jpg')
                pose_keypoints_save_path = os.path.join(pose_keypoints_dir, f'{training_sample_id}.txt')
                cv2.imwrite(person_original_img_save_path, img)
                cv2.imwrite(clothing_img_save_path, clothing_img)
                with open(pose_keypoints_save_path, 'w') as keypoints_file:
                   keypoints_file.write(str(keypoints))     
                if save_for_inspection: 
                    inspection_path_person_original = os.path.join(inspection_dir, f'{training_sample_id}_person_original.jpg')
                    inspection_path_clothing = os.path.join(inspection_dir, f'{training_sample_id}_clothing.jpg')
                    inspection_path_keypoints = os.path.join(inspection_dir, f'{training_sample_id}_keypoints.jpg')
                    cv2.imwrite(inspection_path_person_original, img)
                    cv2.imwrite(inspection_path_clothing, clothing_img)
                    save_img_w_overlaid_keypoints(img, keypoints, inspection_path_keypoints)
                training_sample_num += 1
            
            if training_sample_num > 100:
               return

def preprocess_fto_schp():
    data_source_acronym = 'fto'
    data_source_dir_name = 'fto'
    person_original_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'person_original')
    person_with_masked_clothing_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'person_with_masked_clothing')
    clothing_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'clothing')
    pose_keypoints_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'pose_keypoints')
    mask_coordinates_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'mask_coordinates')
    schp_raw_output_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output')
    inspection_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'inspection')
    original_data_dir = os.path.join(c.ORIGINAL_DATA_DIR, data_source_dir_name)
    log_file_path = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'log.txt')
    
    run_schp(person_original_dir, schp_raw_output_dir)



def preprocess_fto():
#    preprocess_fto_pose()
   preprocess_fto_schp()


preprocess_fto()