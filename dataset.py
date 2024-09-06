import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config

class DogCatDataset(Dataset):
    def __init__(self, root_dir: Path, transform: Optional[Callable] = None):
        self.images_path_label: List[Tuple[Path, int]] = []
        category = {'dogs': 0, 'cats': 1}

        for sub_dir in root_dir.iterdir():
            if sub_dir.is_dir():
                for image_path in sub_dir.glob('*'):
                    self.images_path_label.append((image_path, category[sub_dir.name]))

        self.transform = transform

    def __len__(self) -> int:
        return len(self.images_path_label)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.images_path_label[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label

def get_transform(is_train: bool) -> Callable:
    if is_train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit = config.RANDOM_ROTATION, p = 0.5),
            A.ColorJitter(brightness = config.COLOR_JITTER, contrast = config.COLOR_JITTER, saturation = config.COLOR_JITTER, p=0.5),
            A.Normalize(mean = config.MEAN, std = config.STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean = config.MEAN, std = config.STD),
            ToTensorV2(),
        ])

def get_data_loaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Create datasets
    full_dataset = DogCatDataset(config.TRAIN_PATH, transform = None)
    test_dataset = DogCatDataset(config.TEST_PATH, transform = get_transform(is_train = False))

    train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size = 0.2, random_state = 42)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    # Create data loaders with custom collate functions
    def train_collate_fn(batch):
        transform = get_transform(is_train = True)
        images, labels = zip(*batch)
        images = [transform(image=image)['image'] for image in images]
        return torch.stack(images), torch.tensor(labels)

    def val_collate_fn(batch):
        transform = get_transform(is_train = False)
        images, labels = zip(*batch) 
        images = [transform(image=image)['image'] for image in images]
        return torch.stack(images), torch.tensor(labels)

    train_loader = DataLoader(train_dataset, batch_size = config.BATCH_SIZE, 
                              shuffle = True, num_workers = 4, pin_memory = True, collate_fn = train_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size = config.BATCH_SIZE, 
                              shuffle = False, num_workers = 4, pin_memory = True, collate_fn = val_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = config.BATCH_SIZE, 
                              shuffle = False, num_workers = 4, pin_memory = True)

    print(f"Number of training samples: {len(train_idx)}")
    print(f"Number of validation samples: {len(val_idx)}")
    print(f"Number of test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader