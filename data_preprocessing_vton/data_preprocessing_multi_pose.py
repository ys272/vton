import numpy as np
import os
import cv2
import config as c
from utils import resize_img, count_lines
from data_preprocessing_vton.pose import PoseModel
from data_preprocessing_vton.schp import generate_raw_schp_values, extract_person_without_clothing, extract_clothing, detect_person
from random import random
import multiprocessing
import pickle
from tqdm import tqdm


data_source_acronym = 'multipose'
data_source_dir_name = 'multi_pose'
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
schp_raw_output_dir_lip_person = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output', 'lip_person')
clothing_dir_flat = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'aux', 'flat')
clothing_dir_front = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'aux', 'front')
schp_raw_output_dir_pascal_person = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output', 'pascal_person')
schp_raw_output_dir_atr_front = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output', 'atr_front')
schp_raw_output_dir_atr_person = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output', 'atr_person')
os.makedirs(clothing_dir_flat, exist_ok=True)
os.makedirs(clothing_dir_front, exist_ok=True)
os.makedirs(schp_raw_output_dir_pascal_person, exist_ok=True)
os.makedirs(schp_raw_output_dir_atr_front, exist_ok=True)
os.makedirs(schp_raw_output_dir_atr_person, exist_ok=True)
os.makedirs(schp_raw_output_dir_lip_person, exist_ok=True)


upper_body_clothing_categories = ['Jackets_Vests', 'Shirts_Polos', 'Suiting', 'Sweaters', 'Sweatshirts_Hoodies', 'Tees_Tanks', 'Blouses_Shirts', 'Cardigans', 'Dresses', 'Graphic_Tees', 'Jackets_Coats', 'Rompers_Jumpsuits', 'Sweaters']

def preprocess_pose():
    pose_model = PoseModel()
    processed_dir_numbers = set()
    training_sample_num = 0
    with open(log_pose_file_path, 'w') as log_file:
      for sub_dir_name_1 in ['high res', 'low res']:
          for sub_dir_name_2 in ['MEN', 'WOMEN']:
            absolute_path_sub_dir = os.path.join(original_data_dir, sub_dir_name_1, sub_dir_name_2)
            clothing_category_dir_names = os.listdir(absolute_path_sub_dir)
            for clothing_category_dir_name in clothing_category_dir_names:
                if clothing_category_dir_name in upper_body_clothing_categories:
                    numbered_dir_names = os.listdir(os.path.join(absolute_path_sub_dir, clothing_category_dir_name))
                    for dir_number in tqdm(numbered_dir_names):
                        if sub_dir_name_1 == 'low res':
                          if dir_number in processed_dir_numbers:
                            # If we already encountered this image in the high res folder, don't create another sample.
                            continue
                        else:
                          processed_dir_numbers.add(dir_number)
                        training_sample_dir = os.path.join(absolute_path_sub_dir, clothing_category_dir_name, dir_number)
                        if not os.path.isdir(training_sample_dir):
                            continue
                        training_sample_dir_contents = os.listdir(training_sample_dir)
                        training_sample_sets = {}
                        for filename in training_sample_dir_contents:
                            training_sample_img_type = filename.split('_')[-1].split('.')[0]
                            if training_sample_img_type != 'segment' and training_sample_img_type != 'additional':
                                training_sample_prefix = filename.split('_')[0]
                                training_sample_members = training_sample_sets.setdefault(training_sample_prefix, [])
                                training_sample_members.append(filename)
                        for training_sample_prefix,training_sample_set in training_sample_sets.items():
                            flat_img = None
                            front_img = None
                            for filename in training_sample_set:
                                training_sample_img_type = filename.split('_')[-1].split('.')[0]
                                if training_sample_img_type == 'flat':
                                    flat_img_path = os.path.join(training_sample_dir, filename)
                                    flat_img = resize_img(c.VTON_RESOLUTION['m'][1], c.VTON_RESOLUTION['m'][0], input_img_path=flat_img_path)
                                elif training_sample_img_type == 'front':
                                    front_img_path = os.path.join(training_sample_dir, filename)
                                    front_img = resize_img(c.VTON_RESOLUTION['m'][1], c.VTON_RESOLUTION['m'][0], input_img_path=front_img_path)
                            for filename in training_sample_set:
                                training_sample_img_type = filename.split('_')[-1].split('.')[0]
                                if training_sample_img_type != 'flat' and training_sample_img_type != 'front':
                                    person_img_path = os.path.join(training_sample_dir, filename)
                                    person_img_medium = resize_img(c.VTON_RESOLUTION['m'][1], c.VTON_RESOLUTION['m'][0], input_img_path=person_img_path)
                                    person_img_large = resize_img(c.VTON_RESOLUTION['l'][1], c.VTON_RESOLUTION['l'][0], input_img_path=person_img_path)
                                    keypoints = pose_model.get_keypoints(person_img_medium)
                                    training_sample_id = f'{data_source_acronym}_{dir_number}_{training_sample_num}'
                                    if not keypoints:
                                        log_file.write(f'no keypoints, {dir_number}\n')
                                        save_problematic_data(training_sample_id, person_img_medium)
                                        continue
                                    else:
                                        if front_img is None and flat_img is None:
                                            continue
                                        cv2.imwrite(os.path.join(person_original_dir, 'm', training_sample_id + '.jpg'), person_img_medium)
                                        cv2.imwrite(os.path.join(person_original_dir, 'l', training_sample_id + '.jpg'), person_img_large)
                                        if flat_img is not None:
                                            cv2.imwrite(os.path.join(clothing_dir_flat, training_sample_id + '.jpg'), flat_img)
                                        if front_img is not None:
                                            cv2.imwrite(os.path.join(clothing_dir_front, training_sample_id + '.jpg'), front_img)
                                        pose_keypoints_save_path = os.path.join(pose_keypoints_dir, 'm', f'{training_sample_id}.txt')
                                        with open(pose_keypoints_save_path, 'w') as keypoints_file:
                                            keypoints_file.write(str(keypoints))     
                                            
                                        save_for_inspection = random() < 0.01
                                        if save_for_inspection:
                                            cv2.imwrite(os.path.join(inspection_dir, f'{training_sample_id}_person_original.jpg'), person_img_medium)
                                            if flat_img is not None:
                                                cv2.imwrite(os.path.join(inspection_dir, f'{training_sample_id}_clothing_flat.jpg'), flat_img)
                                            if front_img is not None:
                                                cv2.imwrite(os.path.join(inspection_dir, f'{training_sample_id}_clothing_front.jpg'), front_img)
                                            inspection_path_keypoints = os.path.join(inspection_dir, f'{training_sample_id}_keypoints.jpg')
                                            pose_model.save_or_return_img_w_overlaid_keypoints(person_img_medium, keypoints, output_path=inspection_path_keypoints)
                                        
                                        training_sample_num += 1
                                # if training_sample_num > 1000:
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
        #   logits = np.load(filepath, allow_pickle=True)
        #   argmaxes = np.argmax(logits, axis=-1)
          img_filename = training_sample_id + '.jpg'
          original_img = cv2.imread(os.path.join(person_original_dir, 'm', img_filename))
          if original_img is None:
              continue
          retval = extract_person_without_clothing(filepath, img=original_img, stats=True)
          if retval is None:
            log_file.write(f'no clothing, {filename}\n')
            schp_img = cv2.imread(os.path.join(schp_raw_output_dir_atr_person, training_sample_id+'.png'))
            save_problematic_data(training_sample_id, original_img, schp_img=schp_img)
            continue
            # Delete the data that was saved so far for this training sample, as it cannot be used without the schp masking.
            person_original_img_save_path_large = os.path.join(person_original_dir, 'l', img_filename)
            person_original_img_save_path_medium = os.path.join(person_original_dir, 'm', img_filename)
            clothing_img_save_path = os.path.join(clothing_dir, f'{training_sample_id}.jpg')
            pose_keypoints_save_path = os.path.join(pose_keypoints_dir, 'm', f'{training_sample_id}.txt')
            if os.path.exists(person_original_img_save_path_large):
                os.remove(person_original_img_save_path_large)
            if os.path.exists(person_original_img_save_path_medium):
                os.remove(person_original_img_save_path_medium)
            if os.path.exists(clothing_img_save_path):
                os.remove(clothing_img_save_path)
            if os.path.exists(pose_keypoints_save_path):
                os.remove(pose_keypoints_save_path)
            continue
        
          '''
          Use the "flat" clothing if it exists, otherwise use the "front" image.
          If flat exists, it's ready to be saved as clothing.
          If only front exists, mask out the clothing and save it.
          '''
          clothing_img = None
          clothing_flat_img_path = os.path.join(clothing_dir_flat, training_sample_id + '.jpg')
          if os.path.exists(clothing_flat_img_path):
            clothing_img = cv2.imread(clothing_flat_img_path)
          if clothing_img is None:
            clothing_front_img_path = os.path.join(clothing_dir_front, training_sample_id + '.jpg')
            clothing_img = cv2.imread(clothing_front_img_path)
            filepath = os.path.join(schp_raw_output_dir_atr_front, filename)
            logits = np.load(filepath, allow_pickle=True)
            argmaxes = np.argmax(logits, axis=-1)
            
            clothing_img = extract_clothing(argmaxes, clothing_img)
          
          cv2.imwrite(os.path.join(clothing_dir, 'm', img_filename), clothing_img)
          
          masked_img, mask_coordinates, max_appearing_clothing_type = retval
          masked_img_path = os.path.join(person_with_masked_clothing_dir, 'm', img_filename)
          mask_coordinates_path = os.path.join(mask_coordinates_dir, 'm', training_sample_id + '.npy')
          np.save(mask_coordinates_path, mask_coordinates)
          cv2.imwrite(masked_img_path, masked_img)
          clothing_count[max_appearing_clothing_type] += 1
          
          if training_sample_id+'_person_original.jpg' in saved_for_inspection:
            inspection_path_person_masked = os.path.join(inspection_dir, f'{training_sample_id}_person_masked.jpg')
            cv2.imwrite(inspection_path_person_masked, masked_img)
            inspection_path_clothing = os.path.join(inspection_dir, f'{training_sample_id}_clothing.jpg')
            cv2.imwrite(inspection_path_clothing, clothing_img)
    
    with open(clothing_count_path, 'w') as f:
       f.write(str(clothing_count))


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
    
    print(f'!!! Removing {len(duplicates_to_remove)} training samples !!!')
    img_dirs_to_modify = [person_original_dir, clothing_dir_front, clothing_dir_flat]
    txt_dirs_to_modify = [pose_keypoints_dir]
    for training_sample_id in duplicates_to_remove:
        remove_data(training_sample_id, img_dirs=img_dirs_to_modify, txt_dirs=txt_dirs_to_modify)
        
        
def remove_data(training_sample_id, img_dirs=None, txt_dirs=None):
    path_of_file_to_remove = os.path.join(person_original_dir, 'l', training_sample_id + '.jpg')
    
    if os.path.exists(path_of_file_to_remove):
            os.remove(path_of_file_to_remove)
    for img_dir_to_modify in img_dirs:
        path_of_file_to_remove = os.path.join(img_dir_to_modify, 'm', training_sample_id + '.jpg')
        if os.path.exists(path_of_file_to_remove):
            os.remove(path_of_file_to_remove)
    for txt_dir_to_modify in txt_dirs:
        path_of_file_to_remove = os.path.join(txt_dir_to_modify, 'm', training_sample_id + '.txt')
        if os.path.exists(path_of_file_to_remove):
            os.remove(path_of_file_to_remove)
            
            
def save_problematic_data(training_sample_id, original_img, pose_img = None, schp_img = None):
    original_img_path = os.path.join(problematic_data_dir, training_sample_id + '_original.jpg')
    cv2.imwrite(original_img_path, original_img)
    if pose_img is not None:
        pose_img_path = os.path.join(problematic_data_dir, training_sample_id + '_pose.jpg')
        cv2.imwrite(pose_img_path, pose_img)
    if schp_img is not None:
        schp_img_path = os.path.join(problematic_data_dir, training_sample_id + '_schp.jpg')
        cv2.imwrite(schp_img_path, schp_img)


def filter_non_persons():
    person_original_m_dir = os.path.join(person_original_dir, 'm')
    img_dirs_to_modify = [person_original_dir, clothing_dir_front, clothing_dir_flat, clothing_dir, person_with_masked_clothing_dir]
    txt_dirs_to_modify = [pose_keypoints_dir, mask_coordinates_dir]
    with open(log_person_detection_file_path, 'w') as log_file:
        for filename in tqdm(os.listdir(schp_raw_output_dir_pascal_person)):
            if filename.split('.')[1] != 'npy':
                continue
            filepath = os.path.join(schp_raw_output_dir_pascal_person, filename)
            logits = np.load(filepath, allow_pickle=True)
            argmaxes = np.argmax(logits, axis=-1)
            if not detect_person(argmaxes):
                training_sample_id = filename.split('.')[0]
                log_file.write(f'no person {training_sample_id}\n')
                person_img_medium = cv2.imread(os.path.join(person_original_m_dir, training_sample_id+'.jpg'))
                schp_img = cv2.imread(os.path.join(schp_raw_output_dir_pascal_person, training_sample_id+'.png'))
                try:
                    save_problematic_data(training_sample_id, person_img_medium, schp_img=schp_img)
                except:
                    pass
                # remove_data(training_sample_id, img_dirs=img_dirs_to_modify, txt_dirs=txt_dirs_to_modify)
                
                # Rather than remove everything filter_non_persons flags, only remove the files that have been vetted and saved in a designated file.
                # with open('/home/yoni/Desktop/f/other/delete for multi pose.txt', 'r') as f:
                #     for line in f.readlines():
                #         training_sample_id = line.strip().split(' ')[-1]
                #         person_original_m_dir = os.path.join(person_original_dir, 'm')
                #         img_dirs_to_modify = [person_original_dir, clothing_dir_front, clothing_dir_flat, clothing_dir, person_with_masked_clothing_dir]
                #         txt_dirs_to_modify = [pose_keypoints_dir, mask_coordinates_dir]
                #         remove_data(training_sample_id, img_dirs=img_dirs_to_modify, txt_dirs=txt_dirs_to_modify)


def preprocess():            
    processes = [
    #    multiprocessing.Process(target=preprocess_pose),
    #    multiprocessing.Process(target=remove_duplicates),
    #    multiprocessing.Process(target=generate_raw_schp_values, args=(os.path.join(person_original_dir, 'm'), schp_raw_output_dir_pascal_person), kwargs={'model':'pascal'}),
    #    multiprocessing.Process(target=generate_raw_schp_values, args=(clothing_dir_front, schp_raw_output_dir_atr_front), kwargs={'model':'atr'}),
    #    multiprocessing.Process(target=generate_raw_schp_values, args=(os.path.join(person_original_dir, 'm'), schp_raw_output_dir_atr_person), kwargs={'model':'atr'}),
    # multiprocessing.Process(target=generate_raw_schp_values, args=(os.path.join(person_original_dir, 'm'), schp_raw_output_dir_lip_person), kwargs={'model':'lip'}),
    #    multiprocessing.Process(target=preprocess_schp, args=([4,7],)), # 4,7 is upper-clothes and dress
    #    multiprocessing.Process(target=filter_non_persons),
    ]
    for process in processes:
        process.start()
        process.join()

    # Modify the filenames of the log files to reflect the number of errors they encountered.
    if os.path.isfile(log_pose_file_path):
        num_lines = count_lines(log_pose_file_path)
        log_pose_file_path_new = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, f'log_pose_{num_lines}.txt')
        os.rename(log_pose_file_path, log_pose_file_path_new)

    if os.path.isfile(log_schp_file_path):
        num_lines = count_lines(log_schp_file_path)
        log_schp_file_path_new = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, f'log_schp_{num_lines}.txt')
        os.rename(log_schp_file_path, log_schp_file_path_new)
        
    if os.path.isfile(log_person_detection_file_path):
        num_lines = count_lines(log_person_detection_file_path)
        log_person_detection_file_path_new = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, f'log_person_detection_{num_lines}.txt')
        os.rename(log_person_detection_file_path, log_person_detection_file_path_new)


preprocess()