import cv2
import numpy as np
from typing import Tuple, List


# def resize_image_with_padding(width:int, height:int, input_img_path:str = None, img=None) -> np.ndarray:
#     if img is None:
#         img = cv2.imread(input_img_path)

#     # Calculate the aspect ratio of the original image
#     original_height, original_width = img.shape[:2]
#     aspect_ratio = original_width / original_height

#     # Calculate the new dimensions while maintaining the aspect ratio
#     if width / height > aspect_ratio:
#         new_width = int(height * aspect_ratio)
#         new_height = height
#     else:
#         new_width = width
#         new_height = int(width / aspect_ratio)

#     # Resize the image while maintaining the aspect ratio
#     resized_image = cv2.resize(img, (new_width, new_height))
    
    

#     # Calculate the padding required to achieve the desired dimensions
#     vertical_padding = (height - new_height) // 2
#     horizontal_padding = (width - new_width) // 2

#     # Determine the median color of the image
#     # median_color = np.median(resized_image, axis=(0, 1)).astype(int)
#     # padding_color = tuple(median_color.tolist())
    
#     # white
#     padding_color = (255, 255, 255)

#     # Create a canvas of the desired dimensions with the padding color
#     canvas = np.zeros((height, width, 3), dtype=np.uint8)
#     canvas[:, :] = padding_color

#     # Paste the resized image onto the canvas
#     canvas[vertical_padding:vertical_padding+new_height, horizontal_padding:horizontal_padding+new_width] = resized_image

#     return canvas

# def pad_to_square_and_resize_img(image, target_width, target_height):
#     # Get the original image dimensions
#     height, width = image.shape[:2]

#     # Determine the maximum dimension
#     max_dim = max(width, height)

#     # Calculate the padding required to make the image square
#     pad_x = (max_dim - width) // 2
#     pad_y = (max_dim - height) // 2

#     # Adjust padding if image dimensions are odd
#     if width % 2 != 0:
#         pad_x += 1
#     if height % 2 != 0:
#         pad_y += 1

#     # Create a padded square canvas
#     padded_image = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
#     # Resize the padded image to the target dimensions
#     # resized_image = cv2.resize(padded_image, (target_width, target_height))
#     resized_image = cv2.resize(padded_image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)


#     return resized_image


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
    resized_image = cv2.resize(img, (new_width, new_height))
    
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