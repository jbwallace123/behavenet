""" Dataloader for lightning pytorch implementation """

import os
import json
import numpy as np
import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
    

class ParquetDataset(Dataset):
    def __init__(self, parquet_file, data_type="image", split="train", val_size=0.1, test_size=0.1, height=140, width=170, transform=None, random_seed=42):
        """
        Custom PyTorch dataset for handling Parquet files with train/test/val splits.
        Assumes parquet file has columns:
        - trial: trial number
        - frame_index: frame index within trial
        - neural_data: neural data as a list or numpy array
        - frame_data: image data as bytes, assumes grayscale image e.g. [0, 255]

        Args:
            parquet_file (str): Path to the Parquet file.
            data_type (str): One of 'image' or 'neural' to specify the type of data to load.
            split (str): One of 'train', 'val', or 'test' to specify the dataset partition.
            val_size (float): Proportion of the dataset to use for validation.
            test_size (float): Proportion of the dataset to use for testing.
            height (int): Height of the reshaped image.
            width (int): Width of the reshaped image.
            transform (callable, optional): Optional transform to apply to images.
            random_seed (int): Random seed for reproducibility.
        """
        self.df = pd.read_parquet(parquet_file)
        self.data_type = data_type

        # Split dataset into train, validation, and test sets
        train_val_df, test_df = train_test_split(self.df, test_size=test_size, random_state=random_seed, shuffle=True)
        train_df, val_df = train_test_split(train_val_df, test_size=val_size / (1 - test_size), random_state=random_seed, shuffle=True)

        # Assign appropriate split
        if split == "train":
            self.df = train_df
        elif split == "val":
            self.df = val_df
        elif split == "test":
            self.df = test_df
        else:
            raise ValueError("Invalid split: choose from 'train', 'val', or 'test'.")

        self.height = height
        self.width = width
        self.transform = transform  # Optional: Use for data augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Convert frame_data from bytes to numpy array and reshape
        array = np.frombuffer(row['frame_data'], dtype=np.uint8)
        image = array.reshape((self.height, self.width))

        # Convert image to a PyTorch tensor and normalize
        image = torch.tensor(image, dtype=torch.float32) / 255.0 
        image = image.unsqueeze(0) 

        # Extract metadata
        trial = torch.tensor(row['trial'], dtype=torch.long)
        frame_index = torch.tensor(row['frame_index'], dtype=torch.long)

        # Assumes neural data is binned spike counts
        neural_data = torch.tensor(row['neural_data'], dtype=torch.float32)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        if self.data_type == "image":
            return image, image
        elif self.data_type == "neural":
            return neural_data, neural_data
        else:
            raise ValueError("Invalid data type: choose from 'image' or 'neural'.")