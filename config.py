import os
import torch


ROOT_DIR = '/home/yoni/Desktop/f'
DATA_DIR = os.path.join(ROOT_DIR, 'data')
ORIGINAL_DATA_DIR = os.path.join(DATA_DIR, 'original_data')
DATA_SOURCES = ['same_person_two_poses', 'misc_online', 'misc_world', 'multi_pose', 'paired_low_res', 'paired_high_res']
PREPROCESSED_DATA_VTON_DIR = os.path.join(DATA_DIR, 'processed_data_vton')
PREPROCESSED_DATA_VTON_SUB_DIRS = ['person_original', 'person_with_masked_clothing', 'clothing', 'pose_keypoints', 'mask_coordinates', 'schp_raw_output', 'inspection', 'problematic_data', 'aux']
sml_dirs = ['person_original']
sm_dirs = ['person_with_masked_clothing', 'clothing', 'pose_keypoints', 'mask_coordinates']
dirs = [ROOT_DIR, DATA_DIR, ORIGINAL_DATA_DIR, PREPROCESSED_DATA_VTON_DIR]
for dir in dirs:
    os.makedirs(dir, exist_ok=True)
for dir in DATA_SOURCES:
    for sub_dir in PREPROCESSED_DATA_VTON_SUB_DIRS:
        if sub_dir in sml_dirs:
            for size in ['t', 's', 'm', 'l']:
                os.makedirs(os.path.join(PREPROCESSED_DATA_VTON_DIR, dir, sub_dir, size), exist_ok=True)
        elif sub_dir in sm_dirs:
            for size in ['t', 's', 'm']:
                os.makedirs(os.path.join(PREPROCESSED_DATA_VTON_DIR, dir, sub_dir, size), exist_ok=True)
        else:
            os.makedirs(os.path.join(PREPROCESSED_DATA_VTON_DIR, dir, sub_dir), exist_ok=True)
            
MODEL_OUTPUT_DIR = os.path.join(ROOT_DIR, 'model_output')
MODEL_OUTPUT_PARAMS_DIR = os.path.join(MODEL_OUTPUT_DIR, 'network_params')
MODEL_OUTPUT_TBOARD_DIR = os.path.join(MODEL_OUTPUT_DIR, 'tboard')
MODEL_OUTPUT_IMAGES_DIR = os.path.join(MODEL_OUTPUT_DIR, 'images')
MODEL_OUTPUT_LOG_DIR = os.path.join(MODEL_OUTPUT_DIR, 'logs')
dirs = [MODEL_OUTPUT_DIR, MODEL_OUTPUT_PARAMS_DIR, MODEL_OUTPUT_TBOARD_DIR, MODEL_OUTPUT_IMAGES_DIR, MODEL_OUTPUT_LOG_DIR]
for dir in dirs:
    os.makedirs(dir, exist_ok=True)

READY_DATASETS_DIR = os.path.join(DATA_DIR, 'ready_datasets')

'''
pose model
'''
MOVENET_MODEL_OPTIONS = ['thunder/', # originally from: https://tfhub.dev/google/movenet/singlepose/thunder/4"
                 'lightning/'] # https://tfhub.dev/google/movenet/singlepose/lightning/4
MOVENET_MODEL = MOVENET_MODEL_OPTIONS[0]
keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
idx_to_keypoint_name = {i:keypoint for i,keypoint in enumerate(keypoints)}
keypoint_name_to_idx = {keypoint:i for i,keypoint in enumerate(keypoints)}
idx_to_body_keypoint_name = {i:keypoint for i,keypoint in enumerate(keypoints[5:])}
body_keypoint_name_to_idx = {keypoint:i for i,keypoint in enumerate(keypoints[5:])}
#  {0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear', 5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow', 9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip', 13: 'left_knee', 14: 
# 'right_knee', 15: 'left_ankle', 16: 'right_ankle'}


'''
schp model
'''
SCHP_ROOT_DIR = os.path.join(ROOT_DIR, 'ext-code/Self-Correction-Human-Parsing')
SCHP_SCRIPT_PATH = os.path.join(ROOT_DIR, 'code/algo/scripts/schp.sh')
# atr
schp_labels = ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt', 'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
idx_to_schp_label = {i:label for i,label in enumerate(schp_labels)}
schp_label_to_idx = {label:i for i,label in enumerate(schp_labels)}
# here are the atr values:
# {'Background': 0, 'Hat': 1, 'Hair': 2, 'Sunglasses': 3, 'Upper-clothes': 4, 'Skirt': 5, 'Pants': 6, 
# 'Dress': 7, 'Belt': 8, 'Left-shoe': 9, 'Right-shoe': 10, 'Face': 11, 'Left-leg': 12, 'Right-leg': 13, 
# 'Left-arm': 14, 'Right-arm': 15, 'Bag': 16, 'Scarf': 17}
# here are the pascal values:
# ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs']
# here are the dense pose values:
# 1, 2 = Torso, 3 = Right Hand, 4 = Left Hand, 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right,
# 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right, 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left,
# 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left, 20, 22 = Lower Arm Right, 23, 24 = Head
# here are the lip value:
# {0: 'Background', 1: 'Hat', 2: 'Hair', 3: 'Glove', 4: 'Sunglasses', 5: 'Upper-clothes', 6: 'Dress',
# 7: 'Coat', 8: 'Socks', 9: 'Pants', 10: 'Jumpsuits', 11: 'Scarf', 12: 'Skirt', 13: 'Face', 14: 'Left-arm',
# 15: 'Right-arm', 16: 'Left-leg', 17: 'Right-leg', 18: 'Left-shoe', 19: 'Right-shoe'}
VTON_RESOLUTION = {'t':(64,44), 's':(128,88), 'm':(256,176), 'l':(1024,704)}


MAX_NORMALIZED_VALUE = 1
MIN_NORMALIZED_VALUE = -MAX_NORMALIZED_VALUE

if MIN_NORMALIZED_VALUE == -0.5:
    NOISE_SCALING_FACTOR = 0.5
elif MIN_NORMALIZED_VALUE == -1:
    NOISE_SCALING_FACTOR = 1

REVERSE_DIFFUSION_SAMPLER = 'ddim'
# REVERSE_DIFFUSION_SAMPLER = 'karras'

USE_MIN_SNR_GAMMA_WEIGHTING = False
KARRAS_SIGMA_MAX = 5


'''
General
'''

 # These rarely change.
DEVICE = 'cuda'
RANDOM_SEED = 7
OPTIMIZE = True
USE_BFLOAT16 = True
BATCH_SIZE = 8
MAX_EFFECTIVE_BATCH_SIZE = 64
MAX_ACCUMULATION_RATE = MAX_EFFECTIVE_BATCH_SIZE / BATCH_SIZE
NUM_DIFFUSION_TIMESTEPS = 256

'''
DEBUG
'''
DEBUG_FIND_MIN_MEDIAN_GRAD_PER_BATCH = False


'''
MAIN config vars
'''
USE_CLASSIFIER_FREE_GUIDANCE = True
IMAGE_SIZE = 't'
RUN_EMA = False
EVAL_FREQUENCY = 1010
BATCH_ACCUMULATION = 1
USE_AMP = True
ADAM_EPS = 1e-7 if USE_AMP and not USE_BFLOAT16 else 1e-10 # min value for float16 is approx 6e-8, so epsilon must be larger than that value.

if USE_AMP:
    if USE_BFLOAT16:
        MODEL_DTYPE = torch.bfloat16
    else:
        MODEL_DTYPE = torch.float16
else:
    MODEL_DTYPE = torch.float32
        