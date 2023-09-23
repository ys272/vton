import random
import sys
import os
import cv2
import numpy as np
from typing import Tuple, List
from random import uniform
import config as c


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


def denormalize_img(img):
    if c.MIN_NORMALIZED_VALUE == -0.5:
        img = img + 0.5
    elif c.MIN_NORMALIZED_VALUE == -1:
        img =  (img + 1) * 0.5
    else:
        sys.exit('unsupported normalization')
    return img


COLOR_AQUA = (255, 255, 0) # BGR format
def save_or_return_img_w_overlaid_keypoints(img, keypoint_coords, output_path=None, return_value=False):
    # Draw circles on the image at the given coordinates
    for coord_idx in range(0, len(keypoint_coords), 2):
        x,y = keypoint_coords[coord_idx], keypoint_coords[coord_idx+1]
        if x != 0 or y != 0:
            x = round(x.item() * img.shape[1])
            y = round(y.item() * img.shape[0])
            cv2.circle(img, (x,y), radius=2, color=COLOR_AQUA, thickness=-1)
    if output_path:
        cv2.imwrite(output_path, img)
    if return_value:
        return img
    

def downsample_and_upsample_person(person):
    # Receives and returns a tensor.
    import torch.nn.functional as F
    downsampled_person = F.interpolate(person, size=(c.VTON_RESOLUTION['s'][0], c.VTON_RESOLUTION['s'][1]), mode='area')
    upsampled_person = F.interpolate(downsampled_person, size=(c.VTON_RESOLUTION['m'][0], c.VTON_RESOLUTION['m'][1]), mode='nearest')
    return upsampled_person


def save_tensor_img_to_disk(img, name, target_dir=c.MODEL_OUTPUT_IMAGES_DIR, device=c.DEVICE):
    # img is a tensor.
    import torch
    if device == 'cuda':
        img = (((img.to(dtype=torch.float16).cpu().numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
    else:
        img = (((img.to(dtype=torch.float16).numpy())+1)*127.5).astype(np.uint8)[::-1].transpose(1,2,0)
    cv2.imwrite(os.path.join(target_dir, f'deleteme_{name}.png'), img)


def preprocess_s_person_output_for_m(person, noise_amount_masked):
    # person is a tensor, noise_amount_masked is a number in [0,10000]
    import torch
    noise_tensor = torch.randn_like(person)
    downsampled_person = downsample_and_upsample_person(person)
    noise_amount_masked_scaled = noise_amount_masked / 10000
    inverse_noise_amount_masked_scaled = 1 - noise_amount_masked_scaled
    broadcasted_inverse_noise_amount = inverse_noise_amount_masked_scaled.view(inverse_noise_amount_masked_scaled.shape[0], 1, 1, 1)
    broadcasted_noise_amount = noise_amount_masked_scaled.view(noise_amount_masked_scaled.shape[0], 1, 1, 1)
    noise_augmented_person = downsampled_person * broadcasted_inverse_noise_amount + noise_tensor * broadcasted_noise_amount 
    return noise_augmented_person
    
    # noise_tensor = torch.rand(person.shape[0], 1, person.shape[2], person.shape[3], device=person.device)
    # downsampled_person = person#downsample_and_upsample_person(person)
    # for p_idx in range(len(downsampled_person)):
    #     noise_tensor_p = noise_tensor > 0.9
    #     true_indices = torch.nonzero(noise_tensor_p)
    #     shuffled_indices = true_indices[torch.randperm(true_indices.size(0))]
        
    #     for idx in shuffled_indices:
    #         i, j = idx[0], idx[1]
    #         neighbors = [
    #         downsampled_person[p_idx, :, max(0, i - 1), j], downsampled_person[p_idx, :, min(175, i + 1), j],  # Above and below
    #         downsampled_person[p_idx, :, i, max(0, j - 1)], downsampled_person[p_idx, :, i, min(175, j + 1)]  # Left and right
    #         ]
    #         random_neighbor = random.choice(neighbors)
    #         downsampled_person[p_idx, :, i, j] = random_neighbor
    # return downsampled_person

    
    # noise_amount_masked_scaled = noise_amount_masked / 10000
    # inverse_noise_amount_masked_scaled = 1 - noise_amount_masked_scaled
    # broadcasted_inverse_noise_amount = inverse_noise_amount_masked_scaled.view(inverse_noise_amount_masked_scaled.shape[0], 1, 1, 1)
    # broadcasted_noise_amount = noise_amount_masked_scaled.view(noise_amount_masked_scaled.shape[0], 1, 1, 1)
    # noise_augmented_person = downsampled_person * broadcasted_inverse_noise_amount + noise_tensor * broadcasted_noise_amount 
    # return noise_augmented_person
    
    
    # noise_tensor_pixel = torch.randn(person.shape[0], 1, person.shape[2], person.shape[3], device=person.device)
    # noise_tensor = torch.cat([noise_tensor_pixel] * 3, dim=1)
    # downsampled_person = downsample_and_upsample_person(person)
    # noise_amount_masked_scaled = noise_amount_masked / 10000
    # inverse_noise_amount_masked_scaled = 1 - noise_amount_masked_scaled
    # broadcasted_inverse_noise_amount = inverse_noise_amount_masked_scaled.view(inverse_noise_amount_masked_scaled.shape[0], 1, 1, 1)
    # broadcasted_noise_amount = noise_amount_masked_scaled.view(noise_amount_masked_scaled.shape[0], 1, 1, 1)
    # noise_augmented_person = downsampled_person * broadcasted_inverse_noise_amount + noise_tensor * broadcasted_noise_amount 
    # return noise_augmented_person
