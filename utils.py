import cv2
import numpy as np
from typing import Tuple, List
from random import uniform


def resize_img(target_width:int, target_height:int, input_img_path:str = None, img=None, get_mapping_from_new_to_old_img=False) -> np.ndarray:
    
    if img is None:
        img = cv2.imread(input_img_path)
    
    # Get the original width and height
    original_height, original_width = img.shape[:2]
    
    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height
    
    # Calculate the new width and height based on the aspect ratio
    if target_width / aspect_ratio <= target_height:
        new_width = target_width
        new_height = round(target_width / aspect_ratio)
    else:
        new_width = round(target_height * aspect_ratio)
        new_height = target_height
    
    # Resize the image while maintaining the aspect ratio
    resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create a new blank image with the target width and height
    padded_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    
    # Calculate the position to paste the resized image to center it
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2
    
    # Paste the resized image onto the padded image
    padded_image[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized_image
    
    if get_mapping_from_new_to_old_img:
         # Define the row mapping function
        def row_mapper(row_index):
            if 0 <= row_index - pad_y < new_height:
                return round((row_index - pad_y) * original_height / new_height)
            return None

        # Define the column mapping function
        def col_mapper(col_index):
            if 0 <= col_index - pad_x < new_width:
                return round((col_index - pad_x) * original_width / new_width)
            return None
        return padded_image, row_mapper, col_mapper
    
    return padded_image


def count_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return len(lines)
    