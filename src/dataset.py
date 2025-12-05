import os
import glob
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader



mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(380),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

test_val_transforms = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])



class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        self.loader = default_loader

    def __getitem__(self, idx):
        sample_idx = self.subset.indices[idx]
        path, _ = self.subset.dataset.samples[sample_idx]
        label = self.subset.dataset.targets[sample_idx]
        image = self.loader(path)
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)



class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path



def create_dataloaders(train_dir, test_dir, extra_dir, batch_size=8):

    
    full_dataset = datasets.ImageFolder(train_dir)
    targets = full_dataset.targets

    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, val_test_idx in splitter.split(full_dataset.samples, targets):
        val_size = len(val_test_idx) // 2
        val_idx = val_test_idx[:val_size]
        test_idx = val_test_idx[val_size:]

    
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    valid_subset = torch.utils.data.Subset(full_dataset, val_idx)
    test_subset  = torch.utils.data.Subset(full_dataset, test_idx)

    
    train_dataset = TransformedSubset(train_subset, train_transforms)
    valid_dataset = TransformedSubset(valid_subset, test_val_transforms)
    test_dataset  = TransformedSubset(test_subset, test_val_transforms)

    
    submit_test_dataset = TestDataset(test_dir, transform=test_val_transforms)
    extra_dataset       = TestDataset(extra_dir, transform=test_val_transforms)

   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    submit_test_loader = DataLoader(submit_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    extra_loader = DataLoader(extra_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return {
        "full_dataset": full_dataset,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "submit_test_loader": submit_test_loader,
        "extra_loader": extra_loader,
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "submit_test_dataset": submit_test_dataset,
        "extra_dataset": extra_dataset,
    }
