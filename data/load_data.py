import numpy as np
import torch
from torchvision import datasets, transforms

def get_transforms(input_size):
    '''
    Returns data augmentation transforms in a torch.transforms object.
    '''
    return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
def load_data(data_dir, input_size):
    '''
    Returns a dataset using the transforms listed above. 
    '''
    return datasets.ImageFolder(data_dir, get_transforms(input_size))