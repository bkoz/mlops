import os
import numpy as np
from torch.utils.data import Dataset
# from torchvision.io import read_image

class NpzImageDataset(Dataset):
    def __init__(self, input_file, transform=None, target_transform=None):
        """
        Load a custom dataset stored in numpy compressed format so pytorch
        dataset iterators can be used.
        Args: 
        input_file: string
        transform: A torchvision tranform for the images. (optional)
        target_transform: A torchvision tranform for the labels. (optional)
        """
        self.input_file = input_file
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(self.input_file)['images']
        self.labels = np.load(self.input_file)['labels']
        
    def __len__(self):
        """
        Returns the number of images.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns an image and it's label.
        """
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label