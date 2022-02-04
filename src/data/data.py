import torch
import numpy as np
from npzImageDataset import * 
from torchvision import transforms


def mnist():
    """
    Load the corrupted MNIST training and test datasets.
    Returns train and test loaders.
    """
    #
    # Define a transform to normalize the input data.
    #
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.ConvertImageDtype(torch.float),
                                    transforms.Normalize((0.5,), (0.5,))
                                ])

    trainset = NpzImageDataset(input_file='data/raw/corruptmnist/train_4.npz', transform=transform)
    # print(f'{trainset}') 
    train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = NpzImageDataset(input_file='data/raw/corruptmnist/test.npz', transform=transform)
    # print(f'{testset}') 
    test = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return train, test
