from albumentations.augmentations.transforms import Cutout, HorizontalFlip
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv

from torchvision import transforms

torch.manual_seed(1)


def get_a_train_transform():
    """Get transformer for training data

    Returns:
        Compose: Composed transformations
    """
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=10, p=0.3),
        # A.RandomResizedCrop(height=32, width=32, scale=(0.8, 1.0), p=0.5),
        A.CropAndPad(px=(0, 6), p=0.3, pad_mode=cv.BORDER_REPLICATE),
        A.RandomBrightnessContrast(p=0.3),
        # A.GaussNoise(p=0.2),
        # A.Equalize(p=0.2),
        A.HorizontalFlip(p=0.3),
        A.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1,
                        min_height=8, min_width=8, fill_value=(0.4914, 0.4822, 0.4465), p=0.3),
        ToTensorV2(),
    ])


def get_a_test_transform():
    """Get transformer for test data

    Returns:
        Compose: Composed transformations
    """
    return A.Compose([
        A.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
        ToTensorV2(),
    ])


def get_p_train_transform():
    """Get Pytorch Transform function for train data

    Returns:
        Compose: Composed transformations
    """
    random_rotation_degree = 5
    img_size = (28, 28)
    random_crop_percent = (0.85, 1.0)
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, random_crop_percent),
        transforms.RandomRotation(random_rotation_degree),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])


def get_p_test_transform():
    """Get Pytorch Transform function for test data

    Returns:
        Compose: Composed transformations
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
