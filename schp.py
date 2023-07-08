import numpy as np
import os
import cv2
from config import schp_labels, idx_to_schp_label, schp_label_to_idx, SCHP_ROOT_DIR, SCHP_SCRIPT_PATH
import subprocess
import torch 
import random


BUFFER = 3
def extract_person_without_clothing(argmaxes: np.ndarray, img:np.ndarray = None, clothing_types = [4, 7], stats=False):
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


def extract_person_without_clothing2(filepath_atr: str, img:np.ndarray = None, clothing_types = [4, 7], stats=False):
  '''
  Create a bounding box/rectangle that covers the entire area where the
  clothing currently is (minimum enclosing rectangle).
  '''
  logits = np.load(filepath_atr, allow_pickle=True)
  atr_argmaxes = np.argmax(logits, axis=-1)
  filepath_densepose = '/'.join(filepath_atr.split('/')[:-2]) + '/densepose/' + filepath_atr.split('/')[-1].split('.')[0] + '.pkl'
  with open(filepath_densepose, 'rb') as f:
    densepose_obj = torch.load(f)
  
  num_bboxes_detected = len(densepose_obj['pred_boxes_XYXY'])
  '''
  Use the densepose segmentation results to find the areas of the image we want to keep (hands and head),
  and the areas we want to discard (arms and torso).
  '''
  mask_keep_body = np.zeros((256,176))
  mask_discard_body = np.zeros((256,176))
  for num_bbox in range(num_bboxes_detected):
    # results are ordered as (col,row); which is the opposite of the (row,col) ordering of np/opencv
    bbox = densepose_obj['pred_boxes_XYXY'][num_bbox].cpu()
    row_offset = round(bbox[1].item())
    col_offset = round(bbox[0].item())
    densepose = densepose_obj['pred_densepose'][num_bbox].labels.cpu()
    hands_and_head = np.where(np.isin(densepose, [3,4,23,24]))
    hands_and_head_rows = hands_and_head[0] + row_offset
    hands_and_head_cols = hands_and_head[1] + col_offset
    mask_keep_body[hands_and_head_rows, hands_and_head_cols]=1
    
    torso_and_arms = np.where(np.isin(densepose, [1,2,15,16,17,18,19,20,21,22]))
    torso_and_arms_rows = torso_and_arms[0] + row_offset
    torso_and_arms_cols = torso_and_arms[1] + col_offset
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
  top_y_body = np.min(indices_discard_body[0])
  bottom_y_body = np.max(indices_discard_body[0])
  leftmost_x_body = np.min(indices_discard_body[1])
  rightmost_x_body = np.max(indices_discard_body[1])
  
  top_y_discard = min(top_y_clothing, top_y_body)
  bottom_y_discard = max(bottom_y_clothing, bottom_y_body)
  leftmost_x_discard = min(leftmost_x_clothing, leftmost_x_body)
  rightmost_x_discard = max(rightmost_x_clothing, rightmost_x_body)
  
  top_y_discard = max(0, top_y_discard - buffer)
  bottom_y_discard = min(atr_argmaxes.shape[0] - 1, bottom_y_discard + buffer)
  leftmost_x_discard = max(0, leftmost_x_discard - buffer)
  rightmost_x_discard = min(atr_argmaxes.shape[1] - 1, rightmost_x_discard + buffer)

  mask_discard_final = np.zeros_like(atr_argmaxes, dtype=np.uint8)
  mask_discard_final[top_y_discard:bottom_y_discard, leftmost_x_discard:rightmost_x_discard] = 1

  if img is not None:    
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
  # schp_script_path = os.path.join(SCHP_ROOT_DIR, 'simple_extractor.py')
  # command = ['python', f'{schp_script_path}', '--dataset', f'{model}', '--model-restore', f'{SCHP_ROOT_DIR}/checkpoints/{model}.pth', '--input-dir', f'{input_dir}', '--output-dir', f'{output_dir}', '--logits']
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