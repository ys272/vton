from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import torchvision, torch
from torchvision import transforms
from torch.utils.data import DataLoader
import config as c
import numpy as np


'''
The following code (downloading and saving the fmnist dataset) should be run inside the fmnist virtual env, 
as it requires the most recent version of pytorch.
'''
if False:
  # Step 1: Define the transformation to apply to the dataset
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

  # Step 2: Download the Fashion MNIST dataset
  train_dataset = torchvision.datasets.FashionMNIST(root='/home/yoni/Desktop/f/other/fashion_mnist_data/', train=True, download=True, transform=transform)
  test_dataset = torchvision.datasets.FashionMNIST(root='/home/yoni/Desktop/f/other/fashion_mnist_data/', train=False, download=True, transform=transform)

  # Step 3: Combine train and test datasets and save as an NPZ file
  full_dataset = np.concatenate([train_dataset.data.numpy(), test_dataset.data.numpy()], axis=0)
  full_labels = np.concatenate([train_dataset.targets.numpy(), test_dataset.targets.numpy()], axis=0)

  np.savez('/home/yoni/Desktop/f/other/fashion_mnist_data/fashion_mnist_dataset.npz', data=full_dataset, labels=full_labels)

# Step 4: Load the NPZ file and separate it into train and test sets
loaded_data = np.load('/home/yoni/Desktop/f/other/fashion_mnist_data/fashion_mnist_dataset.npz')
full_dataset = loaded_data['data'].astype(np.float16)
full_labels = loaded_data['labels']

# normalize from [0,255] to [-0.5, 0.5] or [-1, 1].
full_dataset /= 255
full_dataset -= 0.5
if c.MIN_NORMALIZED_VALUE == -1:
  full_dataset *= 2 # [-1,1] normalization

# Assuming you want to split into 80% train and 20% test
num_train = int(0.8 * len(full_dataset))
train_data, train_labels = full_dataset[:num_train], full_labels[:num_train]
test_data, test_labels = full_dataset[num_train:], full_labels[num_train:]

# Convert to PyTorch tensors
train_data = torch.from_numpy(train_data).unsqueeze(1)
train_labels = torch.from_numpy(train_labels).long()
test_data = torch.from_numpy(test_data).unsqueeze(1)
test_labels = torch.from_numpy(test_labels).long()

# Create PyTorch DataLoaders for train and test sets
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=c.BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, test_labels), batch_size=c.BATCH_SIZE, shuffle=False)
