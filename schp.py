import numpy as np
import os
import cv2
from config import schp_labels, idx_to_schp_label, schp_label_to_idx, SCHP_ROOT_DIR, SCHP_SCRIPT_PATH
import subprocess


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

  # orig_img = cv2.imread(basedir2 + f.split('.')[0] + '.jpg')
  # print(basedir + f.split('.')[0] + '_.jpg')
  # cv2.imwrite(basedir + f.split('.')[0] + '_.jpg', orig_img)
  # orig_img[clothing_mask == 1] = [128,128,128]
  # cv2.imwrite(basedir + f.split('.')[0] + '.jpg', orig_img)
  # text = f'{top_y_clothing},{bottom_y_clothing},{leftmost_x_clothing},{rightmost_x_clothing}'
  # with open(basedir + f.split('.')[0] +'.txt', 'w') as file: 
  #   file.write(text)


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

# basedir = r'/home/yoni/Desktop/f/demo/outputs/'
# basedir2 = r'/home/yoni/Desktop/f/demo/inputs/'
# for f in os.listdir(basedir):
#   if f.split('.')[1] == 'npy':
#     filepath = basedir+f
#   else:
#     continue
#   # if f.split('.')[0] != '19_person':
#   #   continue
#   if 'clothing' in f:
#     continue
#   print(f)
#   logits = np.load(filepath)
#   argmaxes = np.argmax(logits, axis=2)
#   extract_person_without_clothing(argmaxes)