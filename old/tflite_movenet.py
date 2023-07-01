pose_sample_rpi_path = '/home/yoni/Desktop/f/tf-lite-movenet/examples/lite/examples/pose_estimation/raspberry_pi'
sys.path.append(pose_sample_rpi_path)

import os, sys, cv2
import numpy as np
from matplotlib import pyplot as plt
import utils
from data import BodyPart
from ml import Movenet
import tensorflow as tf

img_dir = r'/home/yoni/Desktop/f/demo/inputs_orig'
output_dir = r'/home/yoni/Desktop/f/demo/outputs'
movenet = Movenet('/home/yoni/Desktop/f/tf-lite-movenet/movenet_thunder')

# Define function to run pose estimation using MoveNet Thunder.
# You'll apply MoveNet's cropping algorithm and run inference multiple times on
# the input image to improve pose estimation accuracy.
def detect(input_tensor, inference_count=3):
  """Runs detection on an input image.

  Args:
    input_tensor: A [height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.
    inference_count: Number of times the model should run repeatly on the
      same input image to improve detection accuracy.

  Returns:
    A Person entity detected by the MoveNet.SinglePose.
  """
  image_height, image_width, channel = input_tensor.shape

  # Detect pose using the full input image
  movenet.detect(input_tensor.numpy(), reset_crop_region=True)

  # Repeatedly using previous detection result to identify the region of
  # interest and only croping that region to improve detection accuracy
  for _ in range(inference_count - 1):
    person = movenet.detect(input_tensor.numpy(), 
                            reset_crop_region=False)

  return person


def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True,
    keep_input_size=False):
  """Draws the keypoint predictions on image.

  Args:
    image: An numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    person: A person entity returned from the MoveNet.SinglePose model.
    close_figure: Whether to close the plt figure after the function returns.
    keep_input_size: Whether to keep the size of the input image.

  Returns:
    An numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  # Draw the detection result on top of the image.
  image_np = utils.visualize(image, [person])

  # Plot the image with detection results.
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  im = ax.imshow(image_np)

  if close_figure:
    plt.close(fig)

  if not keep_input_size:
    image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))

  return image_np


for img_filename in os.listdir(img_dir):
    print(img_filename)
    # Load the input image.
    image_path = os.path.join(img_dir, img_filename)
    # Save the output image
    # output_image_path = os.path.join(img_dir,f'rescaled_{img_filename}')  # Replace with your desired output image file path
    # cv2.imwrite(output_image_path, resized_image)
    try:
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image)
    except:
        print('Skipped ' + image_path + '. Invalid image.')
        continue
    else:
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image)
        image_height, image_width, channel = image.shape

    # Skip images that isn't RGB because Movenet requires RGB images
    if channel != 3:
        print('Skipped ' + image_path +
                                '. Image isn\'t in RGB format.')
        continue
    # image = tf.io.read_file(input_image_path)
    # image = tf.compat.v1.image.decode_jpeg(image)
    # # image = tf.expand_dims(image, axis=0)

    # new_width = new_height = 192
    # # Resize and pad the image to keep the aspect ratio and fit the expected size.
    # image = tf.cast(tf.image.resize_with_pad(image, new_width, new_height), dtype=tf.int32)

    person = detect(image)
    
    # Save landmarks if all landmarks were detected
    min_landmark_score = min(
        [keypoint.score for keypoint in person.keypoints])
    detection_threshold = 0.1
    should_keep_image = min_landmark_score >= detection_threshold
    if not should_keep_image:
        print('Skipped ' + image_path +
                                '. No pose was confidentlly detected.')
        continue

    # Draw the prediction result on top of the image for debugging later
    output_overlay = draw_prediction_on_image(
        image.numpy().astype(np.uint8), person, 
        close_figure=True, keep_input_size=True)

    # Write detection result into an image file
    output_frame = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, img_filename), output_frame)
