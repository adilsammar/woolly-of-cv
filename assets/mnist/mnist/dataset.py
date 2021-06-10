import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)


class MnistDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.transforms = transforms
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Read Image and Label
        image, label = self.dataset[idx]
        
        image = np.array(image)
        
        # Apply Transforms
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        return (image, label)

    

def get_loader(train_transform, test_transform, batch_size=64, use_cuda=True):
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(
        MnistDataset(datasets.MNIST('../data', train=True, download=True), transforms=train_transform),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(
        MnistDataset(datasets.MNIST('../data', train=False, download=True), transforms=test_transform),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader