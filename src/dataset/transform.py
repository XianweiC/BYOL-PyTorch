import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


# aug: same as MoCov2: https://github.com/facebookresearch/moco/blob/main/main_moco.py#L248
def get_transforms(mode='train', H=32, W=32):
    transform = None
    if mode == 'train':
        transform = A.Compose([
            A.RandomResizedCrop((H, W), scale=(0.08, 1.0), interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            A.ToGray(num_output_channels=3, p=0.2),
            A.GaussianBlur(sigma_limit=(0.1, 2.0), p=0.5),
            A.Solarize(p=0.2),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)), # CIFAR10
            ToTensorV2()
        ])
    elif mode == 'val':
        transform = A.Compose([
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),  # CIFAR10
            ToTensorV2()
        ])
    return transform
