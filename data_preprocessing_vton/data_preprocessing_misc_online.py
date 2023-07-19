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


data_source_acronym = 'misconline'
data_source_dir_name = 'misc_online'
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
schp_raw_output_dir_pascal_person = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output', 'pascal_person')
schp_raw_output_dir_atr_person = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output', 'atr_person')
schp_raw_output_dir_pascal_clothing = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output', 'pascal_clothing')
schp_raw_output_dir_atr_clothing = os.path.join(c.PREPROCESSED_DATA_VTON_DIR, data_source_dir_name, 'schp_raw_output', 'atr_clothing')
os.makedirs(schp_raw_output_dir_pascal_person, exist_ok=True)
os.makedirs(schp_raw_output_dir_atr_person, exist_ok=True)
os.makedirs(schp_raw_output_dir_pascal_clothing, exist_ok=True)
os.makedirs(schp_raw_output_dir_atr_clothing, exist_ok=True)
os.makedirs(schp_raw_output_dir_lip_person, exist_ok=True)


'''
The following function constructs a mapping from raw filenames (stored in d) to clothing categories.
The mapping is stored in d2, which should be used by the preprocessing code.
'''
def map_raw_filenames_to_clothing_categories():
  with open('/home/yoni/Desktop/d.pkl', 'rb') as file:
    d = pickle.load(file)

  cats = list(d.keys())
  cats=[i.lower() for i in cats]
  d2={}
  for i in cats:
    if 'lingerie' in i:
      d2[i]='lingerie'
    elif 'shorts' in i:
      d2[i]='shorts'
    elif 'cardigans' in i:
      d2[i]='cardigans'
    elif 'trousers' in i:
      d2[i]='shorts'
    elif 'jeans' in i:
      d2[i]='pants'
    elif 'underwear' in i:
      d2[i]='underwear'
    elif 'shoes' in i:
      d2[i]='shoes'
    elif 'socks' in i:
      d2[i]='socks'
    elif 'shirt' in i:
      d2[i]='shirts'
    elif 'coats' in i:
      d2[i]='coats'
    elif 'maternity' in i:
      d2[i]='maternity'
    elif 'jewellery' in i:
      d2[i]='jewellry'
    elif 'bags' in i:
      d2[i]='bags'
    elif 'hoodies' in i:
      d2[i]='hoodies'
    elif 'suits' in i:
      d2[i]='suits'
    elif 'hats' in i:
      d2[i]='hats'
    elif 'pyjamas' in i:
      d2[i]='pyjamas'
    elif 'slippers' in i:
      d2[i]='shoes'
    elif 'dresses' in i:
      d2[i]='dresses'
    elif 'bottoms' in i:
      d2[i]='underwear'
    elif 'belts' in i:
      d2[i]='belts'
    elif 'skirts' in i:
      d2[i]='skirts'
    elif 'leggings' in i:
      d2[i]='pants'
    elif 'jacket' in i:
      d2[i]='jackets'
    elif 'ponchos' in i:
      d2[i]='ponchos'
    elif 'blazer' in i:
      d2[i]='blazers'
    elif 'sunglasses' in i:
      d2[i]='sunglasses'
    elif 'swim' in i:
      d2[i]='underwear'
    elif 'scarves' in i:
      d2[i]='scarves'
    elif 'dressing gowns' in i:
      d2[i]='dressing gowns'
    elif 'bra' in i:
      d2[i]='bras'
    elif 'tops' in i:
      d2[i]='tops'
    else:
      d2[i]='other'
#   print('---\n---')
#   for i in cats:
#     if i not in d2:
#       print(i)
#   print(len(d2), len(d)-len(d2), len(set(d2.values())))
#   print(set(d2.values()))
  print(d2)
  

original_clothing_type_to_internal_clothing_type = {'ladies lingerie briefsknickers hipster': 'lingerie', '': 'other', 'ladies swimwear bikinisets tops': 'underwear', 'ladies trousers chinosslacks': 'shorts', 'ladies accessories jewellery earrings': 'jewellry', 'ladies accessories hats caps': 'hats', 'ladies lingerie bras balconette': 'lingerie', 'ladies lingerie briefsknickers bikinibriefs': 'lingerie', 'ladies lingerie bras multipack': 'lingerie', 'ladies lingerie briefsknickers shortiehighwaist': 'lingerie', 'ladies shoes mules': 'shoes', 'ladies nightwear pyjamas': 'pyjamas', 'ladies shoes pumps highheels': 'shoes', 'ladies accessories bags shouldercrossbags': 'bags', 'ladies lingerie bras padded': 'lingerie', 'ladies swimwear swimsuits': 'suits', 'ladies accessories hats buckethats': 'hats', 'ladies accessories hats sunhats': 'hats', 'ladies skirts highwaisted': 'skirts', 'ladies trousers flare': 'shorts', 'ladies dresses shortdresses': 'dresses', 'ladies shirtsblouses blouses': 'shirts', 'ladies dresses bodycon': 'dresses', 'ladies tops vests': 'tops', 'ladies dresses party': 'dresses', 'ladies tops bodys': 'tops', 'ladies tops croppedtops': 'tops', 'ladies basics tops longsleeve': 'tops', 'ladies lingerie bras softwireless': 'lingerie', 'ladies dresses mididresses': 'dresses', 'ladies accessories hairaccessories': 'other', 'ladies accessories belts': 'belts', 'ladies accessories jewellery': 'jewellry', 'ladies lingerie accessories': 'lingerie', 'ladies lingerie bras': 'lingerie', 'ladies nightwear dressing gowns': 'dressing gowns', 'ladies accessories jewellery rings': 'jewellry', 'ladies accessories jewellery necklaces': 'jewellry', 'ladies accessories bags shoppertotes': 'bags', 'ladies maternity lingerietights': 'lingerie', 'ladies maternity tops': 'maternity', 'ladies tops longsleeve': 'tops', 'ladies shoes sandals espandrillos': 'shoes', 'ladies shirtsblouses shirts': 'shirts', 'ladies lingerie briefsknickers multipack': 'lingerie', 'ladies swimwear bikinisets bottoms': 'underwear', 'ladies maternity nursing dresses': 'maternity', 'ladies maternity bottoms': 'maternity', 'ladies shorts highwaisted': 'shorts', 'ladies trousers': 'shorts', 'ladies dresses aline': 'dresses', 'ladies dresses beachdresses': 'dresses', 'sportswear women activity running bra': 'bras', 'sportswear women activity training leggings': 'pants', 'ladies sport sportbras': 'bras', 'sportswear women activity training bra': 'bras', 'ladies sport bottoms': 'underwear', 'ladies trousers wideleg': 'shorts', 'ladies dresses maxidresses': 'dresses', 'ladies tops shortsleeve': 'shorts', 'ladies skirts midiskirts': 'skirts', 'ladies shoes': 'shoes', 'ladies shoes ballerinas flats': 'shoes', 'ladies skirts denim': 'skirts', 'ladies trousers linentrousers': 'shorts', 'ladies basics trousersleggings': 'shorts', 'ladies skirts a-line': 'skirts', 'ladies basics dressesskirts': 'dresses', 'ladies lingerie briefsknickers brazilian': 'lingerie', 'ladies trousers leggings': 'shorts', 'ladies skirts shortskirts': 'shorts', 'ladies divided': 'other', 'ladies basics tops shortsleeve': 'shorts', 'ladies trousers highwaisted': 'shorts', 'ladies trousers cargo': 'shorts', 'ladies blazers': 'blazers', 'ladies trousers culottes': 'shorts', 'ladies tops': 'tops', 'ladies jeans skinny': 'pants', 'ladies accessories bags handbags': 'bags', 'ladies shorts jeans': 'shorts', 'ladies jeans slim': 'pants', 'ladies jeans loose': 'pants', 'ladies shirtblouses offshoulderblouses': 'shirts', 'ladies jumpsuits long': 'suits', 'ladies shirtsblouses tunics': 'shirts', 'ladies tops printed tshirts': 'shirts', 'ladies nightwear loungewear': 'other', 'ladies accessories scarves': 'scarves', 'ladies basics tops vests': 'tops', 'ladies basics tops': 'tops', 'ladies dresses shirt': 'shirts', 'ladies nightwear nightslips': 'other', 'ladies shoes sneakers': 'shoes', 'ladies maternity jeans': 'pants', 'ladies cardigansjumpers jumpers': 'cardigans', 'ladies shorts': 'shorts', 'ladies maternity dresses': 'maternity', 'ladies lingerie': 'lingerie', 'sportswear women activity hiking jacketvest': 'jackets', 'ladies sport bottoms trousers': 'shorts', 'ladies basics cardigansjumpers': 'cardigans', 'sportswear women activity training shorts': 'shorts', 'sportswear women activity running tops': 'tops', 'ladies plus tops': 'tops', 'ladies jacketscoats jackets': 'coats', 'ladies maternity nursing': 'maternity', 'ladies accessories jewellery bracelets': 'jewellry', 'ladies premium selection tops': 'tops', 'ladies trousers joggers': 'shorts', 'ladies premium selection bottoms': 'underwear', 'ladies premium selection': 'other', 'ladies accessories mobileaccessories': 'other', 'ladies accessories sunglasses': 'sunglasses', 'ladies jeans flare': 'pants', 'ladies basics': 'other', 'ladies sockstights socks': 'socks', 'ladies jacketscoats denim': 'coats', 'ladies lingerie bodiescorsets': 'lingerie', 'ladies maternity nursing tops': 'maternity', 'ladies hoodiesswetshirts hoodies': 'shirts', 'ladies jacketscoats bomber': 'coats', 'ladies skirts longskirts': 'skirts', 'ladies shoes heels': 'shoes', 'ladies dresses kaftan': 'dresses', 'kids boys clothing takecare': 'other', 'ladies cardigansjumpers cardigans': 'cardigans', 'ladies hoodiesswetshirts sweatshirts': 'shirts', 'ladies sockstights socks ankle': 'socks', 'sportswear women activity racketsports': 'other', 'sportswear women activity training socks': 'socks', 'sportswear women activity training tops': 'tops', 'ladies tops offshouldertops': 'tops', 'ladies nightwear slippers': 'shoes', 'ladies shoes flipflops': 'shoes', 'ladies premium selection dresses': 'dresses', 'ladies accessories sarongsponchos': 'ponchos', 'ladies dresses lace': 'dresses', 'ladies accessories hats beanies': 'hats', 'ladies basics lingerie': 'lingerie', 'home kitchen napkins': 'other', 'ladies shoes loafers': 'shoes', 'sportswear women activity running leggings': 'pants', 'sportswear women activity running shorts': 'shorts', 'sportswear women activity hiking trousersleggings': 'shorts', 'ladies blazerswaistcoats cropped': 'coats', 'ladies blazerswaistcoats oversized': 'coats', 'home homewear': 'other', 'ladies accessories bags': 'bags', 'ladies jumpsuits short': 'suits', 'ladies jeans bootcut': 'pants', 'ladies shirtsblouses denimshirts': 'shirts', 'men shoes sneakers': 'shoes', 'sportswear men activity training tops': 'tops', 'men trousers cargo': 'shorts', 'sportswear men activity running shorts': 'shorts', 'men hoodiessweatshirts hoodies': 'shirts', 'men hoodiessweatshirts sweatshirts': 'shirts', 'men tshirtstanks shortsleeve': 'shorts', 'men socks multipacks': 'socks', 'men socks ankle': 'socks', 'sportswear men activity training shorts': 'shorts', 'sportswear men activity hiking jacketsvests': 'jackets', 'men shoes dressed': 'shoes', 'men trousers casual': 'shorts', 'men shirts casual': 'shirts', 'men tshirtstanks tanks': 'shirts', 'men shirts longsleeved': 'shirts', 'men shorts chinos': 'shorts', 'men blazerssuits blazers': 'suits', 'men trousers chinos slim all': 'shorts', 'men jeans relaxed': 'pants', 'men shorts casual': 'shorts', 'men shoes sandals espandrillos': 'shoes', 'men shoes loafers': 'shoes', 'men shirt dressed slimfit': 'shirts', 'men socks': 'socks', 'men shirts shortsleeved': 'shorts', 'men tshirtstanks printed': 'shirts', 'men nightwearloungewear pyjamas': 'pyjamas', 'men swimweear': 'underwear', 'men shorts joggers': 'shorts', 'men shorts': 'shorts', 'men trousers joggers': 'shorts', 'men underwear briefs': 'underwear', 'men accessories hatscaps hats': 'hats', 'men underwear trunks': 'underwear', 'men blazerssuits trousers': 'shorts', 'men trousers': 'shorts', 'men shorts cargo': 'shorts', 'men cardigansjumpers turtleneck': 'cardigans', 'men jeans skinny': 'pants', 'men accessories hatscaps caps': 'hats', 'men shorts denim': 'shorts', 'men trousers trousers regular all': 'shorts', 'men tshirtstanks polo': 'shirts', 'sportswear men clothing socks': 'socks', 'men cardigansjumpers jumpers': 'cardigans', 'sportswear men activity racketsports': 'other', 'men jeans regular': 'pants', 'men accessories beltsandbraces': 'belts', 'men trousers linentrousers': 'shorts', 'men accessories bags waistbags': 'bags', 'men trousers dressed': 'shorts', 'men accessories bags shoulderbags': 'bags', 'men shoes': 'shoes', 'men sport bottoms trousers': 'shorts', 'sportswear men activity training hoodies': 'hoodies', 'men tshirtstanks longsleeve': 'shirts', 'men jacketscoats shirtjackets': 'shirts', 'men accessories bags backpack': 'bags', 'men accessories bags': 'bags'}
# unique values in above dict: {'bags', 'maternity', 'skirts', 'shorts', 'scarves', 'pants', 'hoodies', 'tops', 'socks', 'pyjamas', 'bras', 'other', 'jackets', 'hats', 'jewellry', 'dresses', 'cardigans', 'sunglasses', 'underwear', 'suits', 'shirts', 'shoes', 'ponchos', 'coats', 'dressing gowns', 'belts', 'blazers'}
upper_clothes_types = set(['blazers','ponchos', 'coats', 'dressing gowns', 'suits', 'shirts', 'dresses', 'cardigans','jackets','pyjamas', 'tops', 'hoodies', 'lingerie', 'bras'])


def preprocess_pose():
    global d
    pose_model = PoseModel()
    processed_dir_numbers = set()
    training_sample_num = 0
    with open(log_pose_file_path, 'w') as log_file:
        for sub_dir_name in ['ladies', 'men']:
            absolute_path_sub_dir = os.path.join(original_data_dir, sub_dir_name)
            dir_numbers = os.listdir(absolute_path_sub_dir)
            for dir_number in tqdm(dir_numbers):
                dir_number_path = os.path.join(absolute_path_sub_dir, dir_number)
                filenames = os.listdir(dir_number_path)
                filenames = sorted(filenames, key=lambda x: int(x.split('_')[0]))
                # for i in filenames:
                #     extract_clothing_type_from_filename(i)
                # continue
                paired_filenames = []
                i = 0
                while i < len(filenames):
                    numerical_prefix_i = int(filenames[i].split('_')[0])
                    numerical_prefix_next_i = int(filenames[i+1].split('_')[0])
                    if numerical_prefix_i != numerical_prefix_next_i:
                        log_file.write(f'not a pair, {sub_dir_name}, {dir_number}\n')
                        i += 1
                        continue
                    if 'clothing' in filenames[i] and 'person' in filenames[i+1]:
                        paired_filenames.append((filenames[i], filenames[i+1]))
                    elif 'clothing' in filenames[i+1] and 'person' in filenames[i]:
                        paired_filenames.append((filenames[i+1], filenames[i]))
                    else:
                        log_file.write(f'pair missing clothing and person, {sub_dir_name}, {dir_number}, {i}\n')
                    i += 2
                    
                for clothing_filename, person_filename in paired_filenames:
                    clothing_img = cv2.imread(os.path.join(dir_number_path, clothing_filename))
                    clothing_type = extract_clothing_type_from_filename(clothing_filename)
                    training_sample_id = f'{data_source_acronym}_{dir_number}_{training_sample_num}'
                    if clothing_type == '':
                        log_file.write(f'unrecognized category, {sub_dir_name}, {dir_number}, {clothing_filename}\n')
                        save_problematic_data(training_sample_id+f'_{random()}', clothing_img)
                        continue
                    elif clothing_type not in upper_clothes_types:
                        continue
                    person_img_path = os.path.join(dir_number_path, person_filename)
                    person_img_medium = resize_img(c.VTON_RESOLUTION['m'][1], c.VTON_RESOLUTION['m'][0], input_img_path=person_img_path)
                    person_img_large = resize_img(c.VTON_RESOLUTION['l'][1], c.VTON_RESOLUTION['l'][0], input_img_path=person_img_path)
                    clothing_img = resize_img(c.VTON_RESOLUTION['m'][1], c.VTON_RESOLUTION['m'][0], img=clothing_img)
                    keypoints = pose_model.get_keypoints(person_img_medium)
                    if not keypoints:
                        log_file.write(f'no keypoints, {sub_dir_name}, {dir_number}, {person_filename}\n')
                        save_problematic_data(training_sample_id+f'_{random()}', person_img_medium)
                        continue 
                    # Save data
                    cv2.imwrite(os.path.join(person_original_dir, 'm', training_sample_id + '.jpg'), person_img_medium)
                    cv2.imwrite(os.path.join(person_original_dir, 'l', training_sample_id + '.jpg'), person_img_large)
                    cv2.imwrite(os.path.join(clothing_dir, 'm', training_sample_id + '.jpg'), clothing_img)
                    pose_keypoints_save_path = os.path.join(pose_keypoints_dir, 'm', f'{training_sample_id}.txt')
                    with open(pose_keypoints_save_path, 'w') as keypoints_file:
                        keypoints_file.write(str(keypoints))     
                        
                    save_for_inspection = True #random() < 0.02
                    if save_for_inspection:
                        cv2.imwrite(os.path.join(inspection_dir, f'{training_sample_id}_person_original.jpg'), person_img_medium)
                        cv2.imwrite(os.path.join(inspection_dir, f'{training_sample_id}_clothing.jpg'), clothing_img)
                        inspection_path_keypoints = os.path.join(inspection_dir, f'{training_sample_id}_keypoints.jpg')
                        pose_model.save_or_return_img_w_overlaid_keypoints(person_img_medium, keypoints, output_path=inspection_path_keypoints)
                    
                    training_sample_num += 1
            
                    # if training_sample_num > 10:
                      # return
    with open('/home/yoni/Desktop/f/data/processed_data_vton/misc_online/d.pkl', 'wb') as file:
        pickle.dump(d, file)
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
          if not os.path.exists(os.path.join(person_original_dir, 'm', img_filename)):
            continue
          original_img = cv2.imread(os.path.join(person_original_dir, 'm', img_filename))
          # retval = extract_person_without_clothing(argmaxes, img=original_img, stats=True)
          retval = extract_person_without_clothing(filepath, img=original_img, clothing_types=clothing_types, stats=True)
          if retval is None:
            log_file.write(f'no clothing, {filename}\n')
            file_to_remove = os.path.join(schp_raw_output_dir_atr_person, training_sample_id+'.png')
            if os.path.exists(file_to_remove):
              schp_img = cv2.imread(file_to_remove)
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
            else:
                print(f'not exists{clothing_img_save_path}')
            if os.path.exists(pose_keypoints_save_path):
                os.remove(pose_keypoints_save_path)
            continue
        
          # If we successfully extracted the person from the image, save the data.
          masked_img, mask_coordinates, max_appearing_clothing_type = retval
          masked_img_path = os.path.join(person_with_masked_clothing_dir, 'm', img_filename)
          mask_coordinates_path = os.path.join(mask_coordinates_dir, 'm', training_sample_id + '.npy')
          np.save(mask_coordinates_path, mask_coordinates)
          # with open(mask_coordinates_path, 'w') as f:
            #  f.write(str(mask_coordinates))
          cv2.imwrite(masked_img_path, masked_img)
          clothing_count[max_appearing_clothing_type] += 1
          
          # inspection_path_person_masked = os.path.join(inspection_dir, f'{training_sample_id}_person_masked.jpg')
          # cv2.imwrite(inspection_path_person_masked, masked_img)
          # inspection_path_person_masked = os.path.join(inspection_dir, f'{training_sample_id}_person_original.jpg')
          # cv2.imwrite(inspection_path_person_masked, person_img)
          # original_img = cv2.imread(os.path.join(clothing_dir, 'm', img_filename))
          # inspection_path_person_masked = os.path.join(inspection_dir, f'{training_sample_id}_clothing.jpg')
          # cv2.imwrite(inspection_path_person_masked, original_img)
          
          # if training_sample_id+'_person_original.jpg' in saved_for_inspection:
          #   inspection_path_person_masked = os.path.join(inspection_dir, f'{training_sample_id}_person_masked.jpg')
          #   cv2.imwrite(inspection_path_person_masked, masked_img)   
    
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
    img_dirs_to_modify = [person_original_dir, clothing_dir]
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


'''
When processing data downloaded for the first time, the filenames should be saved in d, and then analyzed by the 
function map_raw_filenames_to_clothing_categories, which prints a dictionary that maps the raw filenames
to their clothing types.
'''
d = {}
def extract_clothing_type_from_filename(filename:str) -> str:
    global d
    original_clothing_type = (' '.join(filename.split('_')[2:]).split('.')[0]).lower()
    # d[original_clothing_type] = True
    clothing_type = original_clothing_type_to_internal_clothing_type.get(original_clothing_type, '')
    return clothing_type


def preprocess():            
    processes = [
      #  multiprocessing.Process(target=preprocess_pose),
      #  multiprocessing.Process(target=remove_duplicates),
    #  multiprocessing.Process(target=generate_raw_schp_values, args=(os.path.join(person_original_dir, 'm'), schp_raw_output_dir_atr_person), kwargs={'model':'atr'}),
      #  multiprocessing.Process(target=generate_raw_schp_values, args=(os.path.join(clothing_dir, 'm'), schp_raw_output_dir_atr_clothing), kwargs={'model':'atr'}),
      #  multiprocessing.Process(target=generate_raw_schp_values, args=(os.path.join(person_original_dir, 'm'), schp_raw_output_dir_pascal_person), kwargs={'model':'pascal'}),
      # multiprocessing.Process(target=generate_raw_schp_values, args=(os.path.join(person_original_dir, 'm'), schp_raw_output_dir_lip_person), kwargs={'model':'lip'}),
      #  multiprocessing.Process(target=preprocess_schp, args=([4,7],)), # upper-clothes,dress,coat,jumpsuit
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