import numpy as np
import os
import cv2
import config as c
from utils import resize_img
from data_preprocessing_vton.pose import PoseModel
from data_preprocessing_vton.schp import generate_raw_schp_values, extract_person_without_clothing, detect_person, extract_person_without_clothing_google
from random import random
import multiprocessing
import pickle
from tqdm import tqdm


data_source_acronym = 'pairedhigh'
data_source_dir_name = 'paired_high_res'
person_original_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'person_original')
clothing_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'clothing')
pose_keypoints_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'pose_keypoints')
inspection_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'inspection')
original_data_dir = os.path.join(c.ORIGINAL_DATA_DIR, data_source_dir_name)
person_with_masked_clothing_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'person_with_masked_clothing')
mask_coordinates_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'mask_coordinates')
schp_raw_output_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output')
problematic_data_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'problematic_data')
log_pose_file_path = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'log_pose.txt')
log_schp_file_path = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'log_schp.txt')
log_person_detection_file_path = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'log_person_detection.txt')

schp_raw_output_dir_pascal_person = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output', 'pascal_person')
schp_raw_output_dir_atr_person = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output', 'atr_person')
os.makedirs(schp_raw_output_dir_pascal_person, exist_ok=True)
os.makedirs(schp_raw_output_dir_atr_person, exist_ok=True)


def preprocess_pose():
    pose_model = PoseModel()
    training_sample_num = 0
    with open(log_pose_file_path, 'w') as log_file:
      for sub_dir_name in ['train', 'test']:
        original_data_sub_dir = os.path.join(original_data_dir, sub_dir_name)
        filenames = os.listdir(os.path.join(original_data_sub_dir, 'image'))
        # aa = set(['03721_00.jpg', '10721_00.jpg', '02505_00.jpg'])
        for filename in tqdm(filenames):
            # if filename not in aa:
            #     continue
            person_img_medium = resize_img(c.VTON_RESOLUTION['m'][1], c.VTON_RESOLUTION['m'][0], input_img_path=os.path.join(original_data_sub_dir, 'image', filename))
            person_img_large = resize_img(c.VTON_RESOLUTION['l'][1], c.VTON_RESOLUTION['l'][0], input_img_path=os.path.join(original_data_sub_dir, 'image', filename))
            clothing_img = resize_img(c.VTON_RESOLUTION['m'][1], c.VTON_RESOLUTION['m'][0], input_img_path=os.path.join(original_data_sub_dir, 'cloth', filename))
            training_sample_id = f'{data_source_acronym}_{filename.split(".")[0]}_{training_sample_num}'
            keypoints = pose_model.get_keypoints(person_img_medium)
            if not keypoints:
                log_file.write(f'no keypoints, {filename}\n')
                save_problematic_data(training_sample_id, person_img_medium)
                continue
            save_for_inspection = random() < 0.02
            person_img_medium_save_path = os.path.join(person_original_dir, 'm', f'{training_sample_id}.jpg')
            person_img_large_save_path = os.path.join(person_original_dir, 'l', f'{training_sample_id}.jpg')
            clothing_img_save_path = os.path.join(clothing_dir, 'm', f'{training_sample_id}.jpg')
            pose_keypoints_save_path = os.path.join(pose_keypoints_dir, 'm', f'{training_sample_id}.txt')
            cv2.imwrite(person_img_medium_save_path, person_img_medium)
            cv2.imwrite(person_img_large_save_path, person_img_large)
            cv2.imwrite(clothing_img_save_path, clothing_img)
            with open(pose_keypoints_save_path, 'w') as keypoints_file:
                keypoints_file.write(str(keypoints))     
            if save_for_inspection:
                inspection_path_person_original = os.path.join(inspection_dir, f'{training_sample_id}_person_original.jpg')
                inspection_path_clothing = os.path.join(inspection_dir, f'{training_sample_id}_clothing.jpg')
                inspection_path_keypoints = os.path.join(inspection_dir, f'{training_sample_id}_keypoints.jpg')
                cv2.imwrite(inspection_path_person_original, person_img_medium)
                cv2.imwrite(inspection_path_clothing, clothing_img)
                pose_model.save_or_return_img_w_overlaid_keypoints(person_img_medium, keypoints, output_path=inspection_path_keypoints)
            training_sample_num += 1
            
            # if training_sample_num > 100:
            #     return

    return 
            

def preprocess_schp(clothing_types:list):
    clothing_count_path = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'clothing_count.txt')
    clothing_count = {i:0 for i in clothing_types}

    saved_for_inspection = set()
    for filename in os.listdir(inspection_dir):
        saved_for_inspection.add(filename.split('.')[0])
    
    with open(log_schp_file_path, 'w') as log_file:
        for filename in tqdm(os.listdir(schp_raw_output_dir_atr_person)):
            if filename.split('.')[1] != 'npy':
                continue
            training_sample_id = filename.split('.')[0]
            filepath = os.path.join(schp_raw_output_dir_atr_person, filename)
            # logits = np.load(filepath, allow_pickle=True)
            # argmaxes = np.argmax(logits, axis=-1)
            img_filename = training_sample_id + '.jpg'
            original_img = cv2.imread(os.path.join(person_original_dir, 'm', img_filename))
            retval = extract_person_without_clothing_google(filepath, img=original_img, stats=True)
            if retval is None:
                log_file.write(f'no clothing, {filename}\n')
                schp_img = cv2.imread(os.path.join(schp_raw_output_dir_atr_person, training_sample_id+'.png'))
                save_problematic_data(training_sample_id, original_img, schp_img=schp_img)
                continue
                # Delete the data that was saved so far for this training sample, as it cannot be used without the schp masking.
                # person_original_img_save_path = os.path.join(person_original_dir, 'm', f'{training_sample_id}.jpg')
                # clothing_img_save_path = os.path.join(clothing_dir, 'm', f'{training_sample_id}.jpg')
                # pose_keypoints_save_path = os.path.join(pose_keypoints_dir, 'm', f'{training_sample_id}.txt')
                # os.remove(person_original_img_save_path)
                # os.remove(clothing_img_save_path)
                # os.remove(pose_keypoints_save_path)
            # If we successfully extracted the person from the image, save the data.
            masked_img, mask_coordinates, max_appearing_clothing_type = retval
            masked_img_path = os.path.join(person_with_masked_clothing_dir, 'm', img_filename)
            mask_coordinates_path = os.path.join(mask_coordinates_dir, 'm', training_sample_id + '.npy')
            np.save(mask_coordinates_path, mask_coordinates)
            cv2.imwrite(masked_img_path, masked_img)
            clothing_count[max_appearing_clothing_type] += 1
            
            if training_sample_id+'_person_original.jpg' in saved_for_inspection:
                inspection_path_person_masked = os.path.join(inspection_dir, f'{training_sample_id}_person_masked.jpg')
                cv2.imwrite(inspection_path_person_masked, masked_img)   
    
    with open(clothing_count_path, 'w') as f:
       f.write(str(clothing_count))


def save_problematic_data(training_sample_id, original_img, pose_img = None, schp_img = None):
    original_img_path = os.path.join(problematic_data_dir, training_sample_id + '_original.jpg')
    cv2.imwrite(original_img_path, original_img)
    if pose_img is not None:
        pose_img_path = os.path.join(problematic_data_dir, training_sample_id + '_pose.jpg')
        cv2.imwrite(pose_img_path, pose_img)
    if schp_img is not None:
        schp_img_path = os.path.join(problematic_data_dir, training_sample_id + '_schp.jpg')
        cv2.imwrite(schp_img_path, schp_img)


def count_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return len(lines)


def remove_duplicates():
    sizes = []
    filenames = []
    for filename in tqdm(os.listdir(os.path.join(person_original_dir, 'm'))):
        path = os.path.join(person_original_dir, 'm', filename)
        filenames.append(filename)
        sizes.append(os.path.getsize(path))

    # Sort the lists by size.
    sorted_lists = sorted(zip(sizes, filenames))
    sizes, filenames = zip(*sorted_lists)

    # The first index at which each unique size appears.
    _, first_indices_where_unique_size_appears = np.unique(sizes, return_index=True)
    duplicates_to_remove = set()
    for starting_index_for_this_unique_size in tqdm(range(len(first_indices_where_unique_size_appears[:-1]))):
        # Because the size array (in which we search for unique integers) was sorted, we know that the files in the range
        # below all have the same (compressed) byte size, so we should compare them all to each other for equality.
        for i in range(first_indices_where_unique_size_appears[starting_index_for_this_unique_size], first_indices_where_unique_size_appears[starting_index_for_this_unique_size+1]):
            first_image = cv2.imread(os.path.join(person_original_dir, 'm', filenames[i]))
            # Compare `first_image` to all the images that follow it (that have the same size).
            for j in range(i+1, first_indices_where_unique_size_appears[starting_index_for_this_unique_size+1]):
                second_image = cv2.imread(os.path.join(person_original_dir, 'm', filenames[j]))
                if np.all(first_image == second_image):
                    # If identical, remove the first one.
                    duplicates_to_remove.add(filenames[i].split('.')[0])
                    # No need to search for any further matches for the first image, since it's already been removed.
                    break
    img_dirs_to_modify = [person_original_dir, clothing_dir, person_with_masked_clothing_dir]
    txt_dirs_to_modify = [pose_keypoints_dir, mask_coordinates_dir]
    print(f'!!! Going to remove {len(duplicates_to_remove)} training samples !!!')
    
    for training_sample_id in duplicates_to_remove:
        filename = os.path.join(img_dir_to_modify, 'l', training_sample_id + '.jpg')
        if os.path.exists(filename):
            os.remove(filename)
        for img_dir_to_modify in img_dirs_to_modify:
            filename = os.path.join(img_dir_to_modify, 'm', training_sample_id + '.jpg')
            if os.path.exists(filename):
                os.remove(filename)
        for txt_dir_to_modify in txt_dirs_to_modify:
            filename = os.path.join(txt_dir_to_modify, 'm', training_sample_id + '.txt')
            if os.path.exists(filename):
                os.remove(filename)
            
        
               
def preprocess():
    processes = [
    #    multiprocessing.Process(target=preprocess_pose),
    #    multiprocessing.Process(target=remove_duplicates),
    #    multiprocessing.Process(target=generate_raw_schp_values, args=(os.path.join(person_original_dir, 'm'), schp_raw_output_dir_pascal_person), kwargs={'model':'pascal'}),
    #    multiprocessing.Process(target=generate_raw_schp_values, args=(os.path.join(person_original_dir, 'm'), schp_raw_output_dir_atr_person), kwargs={'model':'atr'}),
       multiprocessing.Process(target=preprocess_schp, args=([4,7],)), # 4,7 is upper-clothes and dress
    ]
    
    for process in processes:
        process.start()
        process.join()

    # Modify the filenames of the log files to reflect the number of errors they encountered.
    if os.path.isfile(log_pose_file_path):
        num_lines_pose_file = count_lines(log_pose_file_path)
        log_pose_file_path_new = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, f'log_pose_{num_lines_pose_file}.txt')
        os.rename(log_pose_file_path, log_pose_file_path_new)

    if os.path.isfile(log_schp_file_path):
        num_lines_schp_file = count_lines(log_schp_file_path)
        log_schp_file_path_new = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, f'log_schp_{num_lines_schp_file}.txt')
        os.rename(log_schp_file_path, log_schp_file_path_new)


preprocess()