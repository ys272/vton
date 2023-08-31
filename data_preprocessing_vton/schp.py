import numpy as np
import os
import cv2
from config import schp_labels, idx_to_schp_label, schp_label_to_idx, SCHP_ROOT_DIR, SCHP_SCRIPT_PATH, VTON_RESOLUTION
import subprocess
import random
import shutil


BUFFER = 3
def extract_person_without_clothing_extra_old(argmaxes: np.ndarray, img:np.ndarray = None, clothing_types = [4,7], stats=False):
  '''
  Create a bounding box/rectangle that covers the entire area where the
  clothing currently is (minimum enclosing rectangle).
  '''
  clothing_pixels = np.where(np.isin(argmaxes, clothing_types))
  if len(clothing_pixels[0]):
    top_y_clothing = np.min(clothing_pixels[0])
    bottom_y_clothing = np.max(clothing_pixels[0])
    leftmost_x_clothing = np.min(clothing_pixels[1])
    rightmost_x_clothing = np.max(clothing_pixels[1])
    if stats:
      detected_clothing_types, counts = np.unique(argmaxes, return_counts=True)
      max_appearing_clothing_type_count = -1
      max_appearing_clothing_type = -1
      for clothing_type in clothing_types:
        idx = np.where(detected_clothing_types==clothing_type)[0]
        if len(idx) > 0:
          idx = idx[0]
          clothing_type_count = counts[idx]
          if clothing_type_count > max_appearing_clothing_type_count:
            max_appearing_clothing_type = clothing_type
            max_appearing_clothing_type_count = clothing_type_count
  else:
    return None
  
  top_y_clothing = max(0, top_y_clothing - BUFFER)
  bottom_y_clothing = min(argmaxes.shape[0] - 1, bottom_y_clothing + BUFFER)
  leftmost_x_clothing = max(0, leftmost_x_clothing - BUFFER)
  rightmost_x_clothing = min(argmaxes.shape[1] - 1, rightmost_x_clothing + BUFFER)

  clothing_mask = np.zeros_like(argmaxes, dtype=np.uint8)
  clothing_mask[top_y_clothing:bottom_y_clothing, leftmost_x_clothing:rightmost_x_clothing] = 1

  if img is not None:
    img[clothing_mask == 1] = [128,128,128]
    mask_coordinates = (top_y_clothing,bottom_y_clothing,leftmost_x_clothing,rightmost_x_clothing)
    if stats: 
      return (img, mask_coordinates, max_appearing_clothing_type)
    else:
      return (img, mask_coordinates)


NUM_PIXELS_COAT_THRESHOLD = 1000
def extract_person_without_clothing(filepath_atr: str, img:np.ndarray = None, clothing_types = [5,6,7,10], stats=False):
  import torch
  
  # Use atr to start by discarding upper clothes and dresses.
  logits = np.load(filepath_atr, allow_pickle=True)
  atr_argmaxes = np.argmax(logits, axis=-1)
  clothing_types = [4,7] # upper clothes and dresses
  mask_discard = np.isin(atr_argmaxes, clothing_types)
  indices_discard_clothing = np.where(mask_discard)
  if len(indices_discard_clothing[0]) == 0:
    return None
  mask_discard = mask_discard.astype(np.uint8)
  # if filepath_atr.split('.')[0].split('/')[-1] == 'misconline_1_304':
  #   print('f')
  
  # Use lip to see if there's a coat present. If it is (and it is more common than the upper clothes, implying that
  # it is probably the target clothing), only mask out the coat, but not the upper clothes.
  # filepath_lip = filepath_atr.replace('atr_person', 'lip_person')
  # logits = np.load(filepath_lip, allow_pickle=True)
  # lip_argmaxes = np.argmax(logits, axis=-1)
  # num_coat_pixels = np.count_nonzero(lip_argmaxes==7)
  # if num_coat_pixels > NUM_PIXELS_COAT_THRESHOLD:
  #   mask_upper_clothes = lip_argmaxes==5
  #   num_upper_clothes_pixels = np.count_nonzero(mask_upper_clothes)
  #   if num_coat_pixels > num_upper_clothes_pixels:
  #     # Do not discard pixels belonging to upper clothes.
  #     mask_discard[mask_upper_clothes] = 0
  # mask_discard[lip_argmaxes==7] = 1
    
  # Erode and dilate the areas we want to discard.
  kernel = np.ones((3, 3), dtype=np.uint8)
  num_dilations = 2 # random.randint(2,5)
  mask_discard = cv2.erode(mask_discard, kernel, iterations= 3)
  mask_discard = cv2.dilate(mask_discard, kernel, iterations= 3 + num_dilations)
  
  # Use the densepose segmentation results to find the hands and make sure they're not removed.  
  filepath_densepose = '/'.join(filepath_atr.split('/')[:-2]) + '/densepose/' + filepath_atr.split('/')[-1].split('.')[0] + '.pkl'
  with open(filepath_densepose, 'rb') as f:
    densepose_obj = torch.load(f)
  num_bboxes_detected = len(densepose_obj['pred_boxes_XYXY'])
  mask_keep_body = np.zeros((VTON_RESOLUTION['m'][0],VTON_RESOLUTION['m'][1]))
  for num_bbox in range(num_bboxes_detected):
    # results are ordered as (col,row); which is the opposite of the (row,col) ordering of np/opencv
    bbox = densepose_obj['pred_boxes_XYXY'][num_bbox].cpu()
    row_offset = round(bbox[1].item())
    col_offset = round(bbox[0].item())
    densepose = densepose_obj['pred_densepose'][num_bbox].labels.cpu()
    hands = np.where(np.isin(densepose, [3,4]))
    hands_rows = hands[0] + row_offset
    hands_cols = hands[1] + col_offset
    hands_rows[hands_rows >= VTON_RESOLUTION['m'][0]] = VTON_RESOLUTION['m'][0]-1
    hands_cols[hands_cols >= VTON_RESOLUTION['m'][1]] = VTON_RESOLUTION['m'][1]-1
    mask_keep_body[hands_rows, hands_cols]=1
  
  atr_w_densepose=[14,15] # left arm, right arm
  mask_keep_body = mask_keep_body.astype(np.uint8)
  mask_keep_body = mask_keep_body & np.isin(atr_argmaxes,atr_w_densepose)    
  
  img[(mask_discard==1) & (mask_keep_body==0)] = [128,128,128]
  
  if stats:
    detected_clothing_types, counts = np.unique(atr_argmaxes, return_counts=True)
    max_appearing_clothing_type_count = -1
    max_appearing_clothing_type = -1
    for clothing_type in clothing_types:
      idx = np.where(detected_clothing_types==clothing_type)[0]
      if len(idx) > 0:
        idx = idx[0]
        clothing_type_count = counts[idx]
        if clothing_type_count > max_appearing_clothing_type_count:
          max_appearing_clothing_type = clothing_type
          max_appearing_clothing_type_count = clothing_type_count
      
  if stats: 
      return (img, mask_discard, max_appearing_clothing_type)
  else:
      return (img, mask_discard)


  


def extract_person_without_clothing_google(filepath_atr: str, img:np.ndarray = None, clothing_types = [4, 7], stats=False):
  import torch
  '''
  Create a bounding box/rectangle that covers the entire area where the
  clothing, torso and arms are (minimum enclosing rectangle).
  The area in the bounding box will be masked out, except for parts of the bounding box that
  overlap with the head, hands, sunglasses, bags and hat.
  '''
  logits = np.load(filepath_atr, allow_pickle=True)
  atr_argmaxes = np.argmax(logits, axis=-1)
  filepath_densepose = '/'.join(filepath_atr.split('/')[:-2]) + '/densepose/' + filepath_atr.split('/')[-1].split('.')[0] + '.pkl'
  # if not os.path.exists(filepath_densepose):
  #   print('***',filepath_densepose)
  #   cv2.imwrite('/home/yoni/Desktop/e/'+filepath_atr.split('/')[-1].split('.')[0] + '.png', img)
  #   return None
  # return None
  with open(filepath_densepose, 'rb') as f:
    densepose_obj = torch.load(f)
  
  num_bboxes_detected = len(densepose_obj['pred_boxes_XYXY'])
  '''
  Use the densepose segmentation results to find the areas of the image we want to keep (hands and head),
  and the areas we want to discard (arms and torso).
  '''
  mask_keep_body = np.zeros((VTON_RESOLUTION['m'][0],VTON_RESOLUTION['m'][1]))
  mask_discard_body = np.zeros((VTON_RESOLUTION['m'][0],VTON_RESOLUTION['m'][1]))
  for num_bbox in range(num_bboxes_detected):
    # results are ordered as (col,row); which is the opposite of the (row,col) ordering of np/opencv
    bbox = densepose_obj['pred_boxes_XYXY'][num_bbox].cpu()
    row_offset = round(bbox[1].item())
    col_offset = round(bbox[0].item())
    densepose = densepose_obj['pred_densepose'][num_bbox].labels.cpu()
    hands_and_head = np.where(np.isin(densepose, [3,4,23,24]))
    hands_and_head_rows = hands_and_head[0] + row_offset
    hands_and_head_cols = hands_and_head[1] + col_offset
    hands_and_head_rows[hands_and_head_rows >= VTON_RESOLUTION['m'][0]] = VTON_RESOLUTION['m'][0]-1
    hands_and_head_cols[hands_and_head_cols >= VTON_RESOLUTION['m'][1]] = VTON_RESOLUTION['m'][1]-1
    mask_keep_body[hands_and_head_rows, hands_and_head_cols]=1
    
    torso_and_arms = np.where(np.isin(densepose, [1,2,15,16,17,18,19,20,21,22]))
    torso_and_arms_rows = torso_and_arms[0] + row_offset
    torso_and_arms_cols = torso_and_arms[1] + col_offset
    torso_and_arms_rows[torso_and_arms_rows >= VTON_RESOLUTION['m'][0]] = VTON_RESOLUTION['m'][0]-1
    torso_and_arms_cols[torso_and_arms_cols >= VTON_RESOLUTION['m'][1]] = VTON_RESOLUTION['m'][1]-1
    mask_discard_body[torso_and_arms_rows, torso_and_arms_cols]=1
      
  mask_keep_body = mask_keep_body.astype(np.uint8)
  mask_discard_body = mask_discard_body.astype(np.uint8)
      
  indices_discard_clothing = np.where(np.isin(atr_argmaxes, clothing_types))
  if len(indices_discard_clothing[0]):
    top_y_clothing = np.min(indices_discard_clothing[0])
    bottom_y_clothing = np.max(indices_discard_clothing[0])
    leftmost_x_clothing = np.min(indices_discard_clothing[1])
    rightmost_x_clothing = np.max(indices_discard_clothing[1])
    if stats:
      detected_clothing_types, counts = np.unique(atr_argmaxes, return_counts=True)
      max_appearing_clothing_type_count = -1
      max_appearing_clothing_type = -1
      for clothing_type in clothing_types:
        idx = np.where(detected_clothing_types==clothing_type)[0]
        if len(idx) > 0:
          idx = idx[0]
          clothing_type_count = counts[idx]
          if clothing_type_count > max_appearing_clothing_type_count:
            max_appearing_clothing_type = clothing_type
            max_appearing_clothing_type_count = clothing_type_count
  else:
    return None
  
  buffer = random.randint(2,4)
  
  indices_discard_body = np.where(mask_discard_body)
  if len(indices_discard_body[0]):
    top_y_body = np.min(indices_discard_body[0])
    bottom_y_body = np.max(indices_discard_body[0])
    leftmost_x_body = np.min(indices_discard_body[1])
    rightmost_x_body = np.max(indices_discard_body[1])
  
    top_y_discard = min(top_y_clothing, top_y_body)
    bottom_y_discard = max(bottom_y_clothing, bottom_y_body)
    leftmost_x_discard = min(leftmost_x_clothing, leftmost_x_body)
    rightmost_x_discard = max(rightmost_x_clothing, rightmost_x_body)
  else:
    top_y_discard = top_y_clothing
    bottom_y_discard = bottom_y_clothing
    leftmost_x_discard = leftmost_x_clothing
    rightmost_x_discard = rightmost_x_clothing
  
  top_y_discard = max(0, top_y_discard - buffer)
  bottom_y_discard = min(atr_argmaxes.shape[0] - 1, bottom_y_discard + buffer)
  leftmost_x_discard = max(0, leftmost_x_discard - buffer)
  rightmost_x_discard = min(atr_argmaxes.shape[1] - 1, rightmost_x_discard + buffer)

  mask_discard_final = np.zeros_like(atr_argmaxes, dtype=np.uint8)
  mask_discard_final[top_y_discard:bottom_y_discard, leftmost_x_discard:rightmost_x_discard] = 1

  # Use lip to see if there's a coat present. If it is (and it is more common than the upper clothes, implying that
  # it is probably the target clothing), only mask out the coat, but not the upper clothes.
  filepath_lip = filepath_atr.replace('atr_person', 'lip_person')
  logits = np.load(filepath_lip, allow_pickle=True)
  lip_argmaxes = np.argmax(logits, axis=-1)
  num_coat_pixels = np.count_nonzero(lip_argmaxes==7)
  if num_coat_pixels > NUM_PIXELS_COAT_THRESHOLD:
    mask_upper_clothes = lip_argmaxes==5
    num_upper_clothes_pixels = np.count_nonzero(mask_upper_clothes)
    if num_coat_pixels > num_upper_clothes_pixels:
      # Do not discard pixels belonging to upper clothes.
      mask_discard_final[mask_upper_clothes] = 0
      
  if img is not None:    
    # TODO: Try adding bag (index=16)
    atr_alone = [1,3] # hat, sunglasses
    # atr_hair = 2
    atr_w_densepose=[11,14,15] # face, left arm, right arm
    kernel = np.ones((3, 3), dtype=np.uint8)
    use_original_values = (np.isin(atr_argmaxes,atr_alone) ) | ((np.isin(atr_argmaxes,atr_w_densepose)) & mask_keep_body)
    use_original_values = use_original_values.astype(np.uint8)
    use_original_values = cv2.dilate(use_original_values, kernel, iterations=5)
    use_original_values = cv2.erode(use_original_values, kernel, iterations=5)
    
    mask_discard_final[use_original_values==1] = 0
    img[mask_discard_final==1] = [128,128,128]
    
    if stats: 
      return (img, mask_discard_final, max_appearing_clothing_type)
    else:
      return (img, mask_discard_final)


def generate_raw_schp_values(input_dir:str, output_dir:str, model:str='atr'):
  command = ['sh', f'{SCHP_SCRIPT_PATH}', SCHP_ROOT_DIR, model, input_dir, output_dir]  
  subprocess.run(command)


def extract_clothing(argmaxes: np.ndarray, img:np.ndarray = None, clothing_types = [4, 7]):
  kernel = np.ones((3, 3), dtype=np.uint8)
  # clothing_mask = np.where(np.isin(argmaxes, clothing_types))
  clothing_mask = np.isin(argmaxes, clothing_types).astype(np.uint8)
  clothing_mask = cv2.dilate(clothing_mask, kernel, iterations=5)
  clothing_mask = cv2.erode(clothing_mask, kernel, iterations=5)
  img[clothing_mask != 1] = [255,255,255]
  return img


def detect_person(argmaxes:np.ndarray) -> bool:
  # A value of 1 corresponds to head. All the pascal values are:
  # ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs']
  return np.count_nonzero(argmaxes==1) > 50

if __name__ == '__main__':
  inputdir = r'/home/yoni/Desktop/input/'
  outputdir = r'/home/yoni/Desktop/pascaloutput/'
  generate_raw_schp_values(inputdir, outputdir, model='pascal')
  # for f in os.listdir(inputdir):
  #   if f.split('.')[1] == 'npy':
  #     filepath = inputdir+f
  #   else:
  #     continue
  #   if 'clothing' in f:
  #     continue
  #   print(f)
  #   logits = np.load(filepath)
  #   argmaxes = np.argmax(logits, axis=2)
  #   extract_person_without_clothing(argmaxes)