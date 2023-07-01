import cv2
import numpy as np



def resize_image_with_padding(width:int, height:int, input_img_path:str = None, img=None) -> np.ndarray:
    if img is None:
        img = cv2.imread(input_img_path)

    # Calculate the aspect ratio of the original image
    original_height, original_width = img.shape[:2]
    aspect_ratio = original_width / original_height

    # Calculate the new dimensions while maintaining the aspect ratio
    if width / height > aspect_ratio:
        new_width = int(height * aspect_ratio)
        new_height = height
    else:
        new_width = width
        new_height = int(width / aspect_ratio)

    # Resize the image while maintaining the aspect ratio
    resized_image = cv2.resize(img, (new_width, new_height))

    # Calculate the padding required to achieve the desired dimensions
    vertical_padding = (height - new_height) // 2
    horizontal_padding = (width - new_width) // 2

    # Determine the median color of the image
    # median_color = np.median(resized_image, axis=(0, 1)).astype(int)
    # padding_color = tuple(median_color.tolist())
    
    # white
    padding_color = (255, 255, 255)

    # Create a canvas of the desired dimensions with the padding color
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = padding_color

    # Paste the resized image onto the canvas
    canvas[vertical_padding:vertical_padding+new_height, horizontal_padding:horizontal_padding+new_width] = resized_image

    return canvas