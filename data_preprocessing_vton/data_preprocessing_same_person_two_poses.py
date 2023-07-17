import numpy as np
import os
import cv2
import config as c
from utils import resize_img
from pose import PoseModel
from schp import generate_raw_schp_values, extract_person_without_clothing, detect_person
from random import random
import multiprocessing
import pickle
from tqdm import tqdm


data_source_acronym = 'sp2p'
data_source_dir_name = 'same_person_two_poses'
person_original_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'person_original')
clothing_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'clothing')
pose_keypoints_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'pose_keypoints')
inspection_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'inspection')
original_data_dir = os.path.join(c.ORIGINAL_DATA_DIR, data_source_dir_name)
person_with_masked_clothing_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'person_with_masked_clothing')
mask_coordinates_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'mask_coordinates')
schp_raw_output_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output')
saved_for_inspection_path = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'saved_for_inspection_path.pkl')
problematic_data_dir = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'problematic_data')
log_pose_file_path = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'log_pose.txt')
log_schp_file_path = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'log_schp.txt')
log_person_detection_file_path = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'log_person_detection.txt')

schp_raw_output_dir_lip_person = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output', 'lip_person')
schp_raw_output_dir_atr_person = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output', 'atr_person')
schp_raw_output_dir_pascal_person = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output', 'pascal_person')
os.makedirs(schp_raw_output_dir_atr_person, exist_ok=True)
os.makedirs(schp_raw_output_dir_pascal_person, exist_ok=True)
os.makedirs(schp_raw_output_dir_lip_person, exist_ok=True)


def preprocess_pose():
    saved_for_inspection = set()
    pose_model = PoseModel()
    training_sample_num = 0
    with open(log_pose_file_path, 'w') as log_file:
      for sub_dir_name in ['train', 'test']:
        original_data_sub_dir = os.path.join(original_data_dir, sub_dir_name)
        numbered_dir_names = os.listdir(original_data_sub_dir)
        for dir_number in tqdm(numbered_dir_names):
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
            '''
            person_img_paths = [img_path for img_path in training_sample_dir_contents if img_path not in set(['mask.jpg', 'target.jpg']) and len(img_path.split('.')) > 1]
            if len(person_img_paths) != 2:
                log_file.write(f'2, {dir_number}, {training_sample_dir_contents}\n')
                continue
            person_imgs_and_keypoints = []
            for person_img_path in person_img_paths:
                person_img_path = os.path.join(training_sample_dir, person_img_path)
                person_img_medium = resize_img(c.VTON_RESOLUTION['m'][1], c.VTON_RESOLUTION['m'][0], input_img_path=person_img_path)
                keypoints = pose_model.get_keypoints(person_img_medium)
                if not keypoints:
                   log_file.write(f'no keypoints, {dir_number}\n')
                   training_sample_id = f'{data_source_acronym}_{dir_number}_{training_sample_num}'
                   save_problematic_data(training_sample_id, person_img_medium)
                else:
                   person_imgs_and_keypoints.append((person_img_medium, keypoints))
            
            clothing_img_path = os.path.join(training_sample_dir, 'target.jpg')
            clothing_img = resize_img(c.VTON_RESOLUTION['m'][1], c.VTON_RESOLUTION['m'][0], input_img_path=clothing_img_path)
            save_for_inspection = random() < 0.02
            for (img,keypoints) in person_imgs_and_keypoints:
                training_sample_id = f'{data_source_acronym}_{dir_number}_{training_sample_num}'
                person_original_img_save_path = os.path.join(person_original_dir, 'm', f'{training_sample_id}.jpg')
                clothing_img_save_path = os.path.join(clothing_dir, 'm', f'{training_sample_id}.jpg')
                pose_keypoints_save_path = os.path.join(pose_keypoints_dir, 'm', f'{training_sample_id}.txt')
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
                    pose_model.save_or_return_img_w_overlaid_keypoints(img, keypoints, output_path=inspection_path_keypoints)
                    saved_for_inspection.add(training_sample_id)
                training_sample_num += 1
            
            # if training_sample_num > 1000:
            #     with open(saved_for_inspection_path, 'wb') as file:
            #         pickle.dump(saved_for_inspection, file)
            #     return saved_for_inspection

    with open(saved_for_inspection_path, 'wb') as file:
        pickle.dump(saved_for_inspection, file)
    return saved_for_inspection
            

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
        #   if training_sample_id != 'sp2p_16599_25':
        #      continue
          filepath = os.path.join(schp_raw_output_dir_atr_person, filename)
        #   logits = np.load(filepath, allow_pickle=True)
        #   argmaxes = np.argmax(logits, axis=-1)
          img_filename = training_sample_id + '.jpg'
          if not os.path.exists(os.path.join(person_original_dir, 'm', img_filename)):
              continue
          original_img = cv2.imread(os.path.join(person_original_dir, 'm', img_filename))
          person_img = np.copy(original_img)
          retval = extract_person_without_clothing(filepath, img=original_img, stats=True)
          if retval is None:
            log_file.write(f'no clothing, {filename}\n')
            # Delete the data that was saved so far for this training sample, as it cannot be used without the schp masking.
            person_original_img_save_path = os.path.join(person_original_dir, 'm', f'{training_sample_id}.jpg')
            clothing_img_save_path = os.path.join(clothing_dir, 'm', f'{training_sample_id}.jpg')
            pose_keypoints_save_path = os.path.join(pose_keypoints_dir, 'm', f'{training_sample_id}.txt')
            continue
            os.remove(person_original_img_save_path)
            os.remove(clothing_img_save_path)
            os.remove(pose_keypoints_save_path)
            schp_img = cv2.imread(os.path.join(schp_raw_output_dir_atr_person, training_sample_id+'.png'))
            save_problematic_data(training_sample_id, original_img, schp_img=schp_img)
            continue
          # If we successfully extracted the person from the image, save the data.
          masked_img, mask_coordinates, max_appearing_clothing_type = retval
          masked_img_path = os.path.join(person_with_masked_clothing_dir, 'm', img_filename)
          mask_coordinates_path = os.path.join(mask_coordinates_dir, 'm', training_sample_id + '.npy')
          np.save(mask_coordinates_path, mask_coordinates)
          cv2.imwrite(masked_img_path, masked_img)
          clothing_count[max_appearing_clothing_type] += 1
          
        #   if random() < 0.01:
        #     inspection_path_person_masked = os.path.join(inspection_dir, f'{training_sample_id}_person_masked.jpg')
        #     cv2.imwrite(inspection_path_person_masked, masked_img)
        #     inspection_path_person_masked = os.path.join(inspection_dir, f'{training_sample_id}_person_original.jpg')
        #     cv2.imwrite(inspection_path_person_masked, person_img)
        #     original_img = cv2.imread(os.path.join(clothing_dir, 'm', img_filename))
        #     inspection_path_person_masked = os.path.join(inspection_dir, f'{training_sample_id}_clothing.jpg')
        #     cv2.imwrite(inspection_path_person_masked, original_img)
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
    # Get the *compressed* size.
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
        for img_dir_to_modify in img_dirs_to_modify:
            os.remove(os.path.join(img_dir_to_modify, 'm', training_sample_id + '.jpg'))
        for txt_dir_to_modify in txt_dirs_to_modify:
            os.remove(os.path.join(txt_dir_to_modify, 'm', training_sample_id + '.txt'))
            
               
def preprocess():
    processes = [
    #    multiprocessing.Process(target=preprocess_pose),
    #    multiprocessing.Process(target=remove_duplicates),
    #    multiprocessing.Process(target=generate_raw_schp_values, args=(os.path.join(person_original_dir, 'm'), schp_raw_output_dir_atr_person)),
    #    multiprocessing.Process(target=generate_raw_schp_values, args=(os.path.join(person_original_dir, 'm'), schp_raw_output_dir_pascal_person), kwargs={'model':'pascal'}),
    #    multiprocessing.Process(target=generate_raw_schp_values, args=(os.path.join(person_original_dir, 'm'), schp_raw_output_dir_lip_person), kwargs={'model':'lip'}),
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