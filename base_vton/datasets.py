import cv2
import os
import config as c
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
      
      
class CustomDataset(Dataset):
    '''
    Provides samples, after applying augmentation (on the fly) to them.
    Each sample is a tuple containing the following:
    clothing_aug, masked_aug, person, pose, sample_original_string_id, sample_unique_ordinal_id, noise_amount_clothing, noise_amount_masked
    '''
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        return self.augment(sample)
    
    def augment(self, sample):
        clothing, masked, person, pose, sample_original_string_id, sample_unique_ordinal_id = sample
        noise_amount_clothing = np.random.rand() / 10
        noise_tensor = torch.randn_like(clothing)
        clothing_aug = clothing * (1 - noise_amount_clothing) + noise_tensor * noise_amount_clothing
        
        noise_amount_masked = np.random.rand() / 10
        noise_tensor = torch.randn_like(masked)
        masked_aug = masked * (1 - noise_amount_masked) + noise_tensor * noise_amount_masked
        # return the sample, replacing the original clothing and masked images with their augmented versions, 
        # and adding the noise amounts (scaled by 10, so that they'll be [0,1]).
        augmented_sample = (clothing_aug, masked_aug, person, pose, sample_original_string_id, sample_unique_ordinal_id, (noise_amount_clothing*10), (noise_amount_masked*10))
        
        # demo
        # img = ((clothing_aug+1)*127.5).cpu().numpy().astype(np.uint8)
        # img_ = ((clothing+1)*127.5).cpu().numpy().astype(np.uint8)
        # cv2.imwrite(f'/home/yoni/Desktop/examples/{noise_amount_clothing*10}.jpg', img)
        # cv2.imwrite(f'/home/yoni/Desktop/examples/{noise_amount_clothing*10}_.jpg', img_)
        
        return augmented_sample  


def process_keypoints(keypoints):
    normalized_keypoints = []
    for keypoint in keypoints:
        if keypoint is None:
            normalized_keypoints.append(-1)
            normalized_keypoints.append(-1)
        else:
            normalized_keypoints.append(keypoint[0]/width)
            normalized_keypoints.append(keypoint[1]/height)
            assert keypoint[0]/width <= 1
            assert keypoint[1]/height <= 1
    normalized_keypoints = torch.tensor(normalized_keypoints, dtype=torch.float16)
    return normalized_keypoints


size = 't'
dataset_dir = os.path.join(c.READY_DATASETS_DIR, f'vton_{size}_to_{size}')
height = c.VTON_RESOLUTION[size][0]
width = c.VTON_RESOLUTION[size][1]
all_samples = {}

filenames = os.listdir(dataset_dir)
random.seed(c.RANDOM_SEED)
random.shuffle(filenames)

for filename in filenames:
    # e.g 'misconline_72_332', shared between orig and aug, if aug exists
    sample_original_string_id = '_'.join(filename.split('_')[:3])
    # e.g '1', unique for every sample
    sample_unique_ordinal_id = int(filename.split('_')[-3])
    sample_type = filename.split('_')[-1].split('.')[0]
    file_type = filename.split('.')[-1]
    if file_type == 'pth':
        sample_data = torch.load(os.path.join(dataset_dir, filename))
    else:
        with open(os.path.join(dataset_dir, filename), 'r') as f:
            sample_data = process_keypoints(eval(f.readlines()[0]))
    sample = all_samples.setdefault(sample_original_string_id, [])
    sample.append((sample_data, sample_original_string_id, sample_unique_ordinal_id, sample_type))

train_frac = 0.92
val_frac = 0.04
test_frac = 1 - train_frac - val_frac

num_total_samples = len(os.listdir(dataset_dir)) / 4
num_required_train_samples = int(train_frac * num_total_samples)
num_required_val_samples = int(val_frac * num_total_samples)
num_required_test_samples = num_total_samples - num_required_train_samples - num_required_val_samples

train_samples = []
val_samples = []
test_samples = []

num_added_train_samples = 0
num_added_val_samples = 0
num_added_test_samples = 0
for sample_original_string_id, sample in all_samples.items():
    num_samples = len(sample) / 4
    sample.sort(key=lambda x:(x[2],x[3]))
    # clothing, masked, person, pose, sample_original_string_id, sample_unique_ordinal_id
    final_sample_orig = (sample[0][0], sample[1][0], sample[2][0], sample[3][0], sample[0][1], sample[0][2])
    if num_samples == 2:
        final_sample_aug = (sample[4][0], sample[5][0], sample[6][0], sample[7][0], sample[4][1], sample[4][2])
    if num_added_train_samples < num_required_train_samples:
        num_added_train_samples += num_samples
        train_samples.append(final_sample_orig)
        if num_samples == 2:
            train_samples.append(final_sample_aug)
    elif num_added_val_samples < num_required_val_samples:
        num_added_val_samples += num_samples
        val_samples.append(final_sample_orig)
        if num_samples == 2:
            val_samples.append(final_sample_aug)
    else:
        num_added_test_samples += num_samples
        test_samples.append(final_sample_orig)
        if num_samples == 2:
            test_samples.append(final_sample_aug)
    
    
print(f'# samples in train, val and test: {num_added_train_samples}, {num_added_val_samples}, {num_added_test_samples}\n')
print(f'% samples in train, val and test: {num_added_train_samples/num_total_samples}, {num_added_val_samples/num_total_samples}, {num_added_test_samples/num_total_samples}\n')


train_dataset = CustomDataset(train_samples)
valid_dataset = CustomDataset(val_samples)
test_dataset = CustomDataset(test_samples)

# Set batch size and other options as needed
train_dataloader = DataLoader(train_dataset, batch_size=c.BATCH_SIZE, shuffle=True, num_workers=3, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=c.BATCH_SIZE, shuffle=False, num_workers=3, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=c.BATCH_SIZE, shuffle=False)


# for batch in train_dataloader:
#     pass

# for batch in valid_dataloader:
#     pass

# for batch in test_dataloader:
#     pass
