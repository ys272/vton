import cv2
import os
import config as c
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import time
from tqdm import tqdm

      
class CustomDataset(Dataset):
    '''
    Provides samples, after applying augmentation (on the fly) to them.
    Each sample is a tuple containing the following:
    clothing_aug, masked_aug, person, pose, sample_original_string_id, sample_unique_string_id, noise_amount_clothing, noise_amount_masked
    '''
    def __init__(self, data_list):
        self.data_list = data_list
        self.height = c.VTON_RESOLUTION[c.IMAGE_SIZE][0]
        self.width = c.VTON_RESOLUTION[c.IMAGE_SIZE][1]
        self.max_height = self.height - 1
        self.max_width = self.width - 1

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        clothing, mask_coords, masked, person, pose_vector, sample_original_string_id, sample_unique_string_id = sample
        # There are a total of 17 keypoints, but the first five are of the face rather than the body.
        # For the concatenated keypoints, we only use the body keypoints (in the vector we use everything).
        num_needed_keypoint_dims = 12
        pose_matrix = torch.zeros((num_needed_keypoint_dims, self.height, self.width), dtype=c.MODEL_DTYPE)
        # Thje vector flattened the pairs to a single 1D list, so the first 5 keypoints pairs now take 10 elements in total.
        for p_idx in range(10, len(pose_vector), 2):
            # We flip the order of the keypoints because pytorch and tensorflow (where the keypoints come from) use a different axis ordering system.
            y = pose_vector[p_idx]
            x = pose_vector[p_idx+1]
            if x==0 and y==0:
                continue
            x = torch.round(x * self.height)
            y = torch.round(y * self.width)
            pose_matrix[int((p_idx - 10)/2)][int(min(self.max_height, x)), int(min(self.max_width, y))] = 1
        
        # unaugmented_sample = (clothing, mask_coords, masked, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id, 0, 0)
        # return unaugmented_sample
        
        unaugmented_sample = (clothing, mask_coords, masked, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id)
        augmented_sample = self.augment(unaugmented_sample)
        return augmented_sample


    def augment(self, sample):
        clothing, mask_coords, masked, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id = sample
        noise_amount_clothing = np.random.rand() / 10
        noise_tensor = torch.randn_like(clothing)
        clothing_aug = clothing * (1 - noise_amount_clothing) + noise_tensor * noise_amount_clothing
        
        noise_amount_masked = np.random.rand() / 10
        noise_tensor = torch.randn_like(masked)
        masked_aug = masked * (1 - noise_amount_masked) + noise_tensor * noise_amount_masked
        masked_aug[:, mask_coords] = masked[:, mask_coords]
        # return the sample, replacing the original clothing and masked images with their augmented versions, 
        # and adding the noise amounts (scaled by 10, so that they'll be [0,1]).
        augmented_sample = (clothing_aug, mask_coords, masked_aug, person, pose_vector, pose_matrix, sample_original_string_id, sample_unique_string_id, int(noise_amount_clothing*10000), int(noise_amount_masked*10000))
        # demo
        # img = ((clothing_aug+1)*127.5).cpu().to(torch.float16).numpy().astype(np.uint8).transpose(1,2,0)
        # img_ = ((clothing+1)*127.5).cpu().to(torch.float16).numpy().astype(np.uint8).transpose(1,2,0)
        # cv2.imwrite(f'/home/yoni/Desktop/examples/{noise_amount_clothing*10}.jpg', img)
        # cv2.imwrite(f'/home/yoni/Desktop/examples/{noise_amount_clothing*10}_.jpg', img_)        
        return augmented_sample  


def create_datasets():
    
    def process_keypoints(keypoints):
        ''' 
        Normalize keypoints array from integer coordinates to values in [0, 1].
        If a coordinate is missing (the model was unconfident regarding its location),
        it should be replaced with (0,0).
        '''
        normalized_keypoints = []
        for keypoint in keypoints:
            if keypoint is None:
                normalized_keypoints.append(0)
                normalized_keypoints.append(0)
            else:
                normalized_keypoints.append(keypoint[0]/width)
                normalized_keypoints.append(keypoint[1]/height)
                assert keypoint[0]/width <= 1
                assert keypoint[1]/height <= 1
        normalized_keypoints = torch.tensor(normalized_keypoints, dtype=c.MODEL_DTYPE)
        return normalized_keypoints

    start_time = time.time()
    print(f'Started loading data')

    size = c.IMAGE_SIZE
    dataset_dir = os.path.join(c.READY_DATASETS_DIR, f'vton_{size}_to_{size}')
    height = c.VTON_RESOLUTION[size][0]
    width = c.VTON_RESOLUTION[size][1]

    all_samples = {}

    filenames = os.listdir(dataset_dir)
    random.seed(c.RANDOM_SEED)
    random.shuffle(filenames)

    '''
    For each filename, we save metadata allowing us to:
    1. Sort the filenames and divide them between train/val/test sets such that augmented versions of 
    the same images aren't split across sets. This is achieved by inserting the (metadata of the) filenames 
    into a dictionary, where each key points to a list containing the metadata of the (one or two) filenames 
    belonging to the original and augmented (if it exists) copy/ies of an image.
    2. Determine the original locations of the files later on.
    '''
    for filename in tqdm(filenames):
        # e.g 'misconline_72_332', shared between orig and aug, if aug exists
        sample_original_string_id = '_'.join(filename.split('_')[:3])
        # e.g '1_aug'.  unique for every sample.
        sample_unique_string_id = '_'.join(filename.split('_')[-3:-1])
        # e.g 'clothing'
        sample_type = filename.split('_')[-1].split('.')[0]
        # e.g 'pth' or 'txt'
        file_type = filename.split('.')[-1]
        if file_type == 'pth':
            sample_data = torch.load(os.path.join(dataset_dir, filename))
        else:
            with open(os.path.join(dataset_dir, filename), 'r') as f:
                sample_data = process_keypoints(eval(f.readlines()[0]))
        sample = all_samples.setdefault(sample_original_string_id, [])
        sample.append((sample_data, sample_original_string_id, sample_unique_string_id, sample_type))

    train_frac = 0.97
    val_frac = 0.01
    test_frac = 1 - train_frac - val_frac
    # train_frac = 0.1
    # val_frac = 0.01
    # test_frac = 1 - train_frac - val_frac

    num_files_per_sample = 5 # clothing, masked person, original person, pose keypoints, mask coordinates
    num_total_samples = len(os.listdir(dataset_dir)) / num_files_per_sample
    num_required_train_samples = int(train_frac * num_total_samples)
    num_required_val_samples = int(val_frac * num_total_samples)
    num_required_test_samples = num_total_samples - num_required_train_samples - num_required_val_samples

    train_samples = []
    val_samples = []
    test_samples = []

    num_added_train_samples = 0
    num_added_val_samples = 0
    num_added_test_samples = 0
    for sample_original_string_id, sample in tqdm(all_samples.items()):
        num_samples = len(sample) / num_files_per_sample
        # sort by the unique ordinal id (x[2]) of each sample. This will separate the original and augmented
        # file versions. Then, the different files belonging to each (original or augmented) sample
        # should be sorted by the 'sample type', so that we have a consistent ordering.
        sample.sort(key=lambda x:(x[2],x[3]))
        # clothing, mask-coords, masked, person, pose vector, sample_original_string_id, sample_unique_string_id
        final_sample_orig = (sample[0][0], sample[1][0], sample[2][0], sample[3][0], sample[4][0], sample[0][1], sample[0][2])
        if num_samples == 2:
            final_sample_aug = (sample[5][0], sample[6][0], sample[7][0], sample[8][0], sample[9][0], sample[5][1], sample[5][2])
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
    print(f'% samples in train, val and test: {(num_added_train_samples/num_total_samples):.2f}, {(num_added_val_samples/num_total_samples):.2f}, {(num_added_test_samples/num_total_samples):.2f}\n')

    train_dataset = CustomDataset(train_samples)
    valid_dataset = CustomDataset(val_samples)
    test_dataset = CustomDataset(test_samples)

    # Set batch size and other options as needed
    train_dataloader = DataLoader(train_dataset, batch_size=c.BATCH_SIZE, shuffle=True, num_workers=3, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=c.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=c.BATCH_SIZE, shuffle=False)

    # torch.save(test_dataloader, f'/home/yoni/Desktop/f/data/ready_datasets/test_dataloader_{size}.pth')

    end_time = time.time()
    print(f'Finished loading data: {end_time-start_time}')
    
    return train_dataloader, valid_dataloader, test_dataloader

    for batch in train_dataloader:
        pass

    for batch in valid_dataloader:
        pass

    for batch in test_dataloader:
        pass


if __name__ == '__main__':
    create_datasets()