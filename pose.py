# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import os 
from utils import resize_img
from config import keypoints, idx_to_keypoint_name, keypoint_name_to_idx, MOVENET_MODEL
from typing import Union


if MOVENET_MODEL.startswith('thunder'):
        resolution = 256
else:
    resolution = 192

class PoseModel:
    def __init__(self):
        self.model = hub.load('/home/yoni/Desktop/f/ext-code/movenet/' + MOVENET_MODEL)
        self.movenet = self.model.signatures['serving_default']
        self.COLOR_AQUA = (255, 255, 0) # BGR format

    def is_person_detected(self, img) -> bool:      
        PERSON_DETECTION_CONFIDENCE_THRESHOLD = 0.75
        # Return False if a person wasn't detected, else - return True
        if img.shape[0] != resolution or img.shape[1] != resolution:
            img = resize_img(resolution, resolution, img=img)
        img = tf.cast(img, dtype=tf.int32)
        img = tf.expand_dims(img, axis=0)
        outputs = self.movenet(img)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints = outputs['output_0']
        # List of coordinates (x, y) where you want to draw circles
        coords = [k for k in keypoints[0][0].numpy() if k[2] > PERSON_DETECTION_CONFIDENCE_THRESHOLD]
        if len(coords) < 5:
            return False
        return True


    def get_keypoints(self, img, num_points=5) -> Union[bool, np.ndarray]:
        PERSON_DETECTION_CONFIDENCE_THRESHOLD = 0.25
        # Return False if a person isn't detected. Otherwise, return keypoints.
        row_mapper = col_mapper = None
        if img.shape[0] != resolution or img.shape[1] != resolution:
            img, row_mapper, col_mapper = resize_img(resolution, resolution, img=img, get_mapping_from_new_to_old_img=True)
        img = tf.cast(img, dtype=tf.int32)
        img = tf.expand_dims(img, axis=0)
        outputs = self.movenet(img)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints = outputs['output_0']
        coords = []
        confident_coords = 0
        for k in keypoints[0][0].numpy():
            if k[2] > PERSON_DETECTION_CONFIDENCE_THRESHOLD:
                row_idx = k[0]*resolution
                col_idx = k[1]*resolution
                original_row_idx = row_mapper(row_idx) if row_mapper else row_idx
                original_col_idx = col_mapper(col_idx) if col_mapper else col_idx
                if original_row_idx is None or original_col_idx is None:
                    coords.append(None)
                else:
                    confident_coords += 1
                    coords.append((int(original_col_idx), int(original_row_idx)))
            else:
                coords.append(None) 
        if confident_coords < num_points:
            return False
        return coords


    def save_or_return_img_w_overlaid_keypoints(self, img, keypoint_coords, output_path=None, return_value=False):
        # This assumes the image is a square
        # scale_factor_0 = img.shape[0] / resolution
        # scale_factor_1 = img.shape[1] / resolution
        # Draw circles on the image at the given coordinates
        for coord in keypoint_coords:
            if coord is not None:
                # coord = (int(coord[0] * scale_factor_0), int(coord[1] * scale_factor_1))
                cv2.circle(img, coord, radius=2, color=self.COLOR_AQUA, thickness=-1)
        # Save the resulting image with drawn circles
        if output_path:
            cv2.imwrite(output_path, img)
        if return_value:
            return img
        


if __name__ == '__main__':
    img_dir = r'/home/yoni/Desktop/f/demo/inputs'
    output_dir = r'/home/yoni/Desktop/f/demo/outputs'        
    new_width = resolution
    new_height = resolution

    for img_filename in os.listdir(img_dir):
        # Load the input image.
        input_image_path = os.path.join(img_dir, img_filename)

        resized_image = resize_img(new_width, new_height, input_img_path=input_image_path)
        # Save the output image
        # output_image_path = os.path.join(img_dir,f'rescaled_{img_filename}')  # Replace with your desired output image file path
        # cv2.imwrite(output_image_path, resized_image)

        image = tf.io.read_file(input_image_path)
        image = tf.compat.v1.image.decode_jpeg(image)
        image = tf.expand_dims(image, axis=0)

        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        image = tf.cast(tf.image.resize_with_pad(image, new_width, new_height), dtype=tf.int32)
        # Run model inference.
        pose_model = PoseModel()
        outputs = pose_model.movenet(image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints = outputs['output_0']
        # List of coordinates (x, y) where you want to draw circles
        THRESHOLD = 0.5
        coords = [(int(k[1]*resolution),int(k[0]*resolution)) for k in keypoints[0][0].numpy() if k[2] > THRESHOLD]
        if len(coords) < 5:
            continue
        print(img_filename, len(coords))
        # Define the color (Aqua in BGR format)
        color = (255, 255, 0)
        # Draw circles on the image at the given coordinates
        # breakpoint()
        for coord in coords:
            cv2.circle(resized_image, coord, radius=2, color=color, thickness=-1)
        output_image_path = os.path.join(output_dir, f'circles_{img_filename}')
        # Save the resulting image with drawn circles
        cv2.imwrite(output_image_path, resized_image)