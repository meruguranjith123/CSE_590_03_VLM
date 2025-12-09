"""
Dataset loaders for CIFAR-100, ImageNet-10k, and COCO
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
from PIL import Image
import json


class CIFAR100Dataset(Dataset):
    """CIFAR-100 Dataset"""
    def __init__(self, root: str, train: bool = True, transform=None):
        self.dataset = datasets.CIFAR100(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Convert to tensor if not already
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        # Convert to [0, 255] range
        image = (image * 255).long().clamp(0, 255)
        return image, label


class ImageNet10kDataset(Dataset):
    """ImageNet-10k Dataset (subset of ImageNet)"""
    def __init__(self, root: str, split: str = 'train', transform=None, max_samples: int = 10000):
        self.root = root
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        
        # ImageNet structure: root/train/class/image.jpg or root/val/class/image.jpg
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"ImageNet split directory not found: {split_dir}")
        
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        classes = sorted(os.listdir(split_dir))
        for class_idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = class_idx
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                images = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for img_name in images[:max_samples // len(classes) + 1]:
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
        
        if len(self.images) > max_samples:
            self.images = self.images[:max_samples]
            self.labels = self.labels[:max_samples]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Convert to [0, 255] range
        if isinstance(image, torch.Tensor):
            if image.max() <= 1.0:
                image = (image * 255).long().clamp(0, 255)
        else:
            image = transforms.ToTensor()(image)
            image = (image * 255).long().clamp(0, 255)
        
        return image, label


class COCODataset(Dataset):
    """COCO Dataset for image generation"""
    def __init__(self, root: str, split: str = 'train', transform=None, max_samples: int = None):
        self.root = root
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        
        # COCO structure: root/images/train2017/ or root/images/val2017/
        image_dir = os.path.join(root, 'images', f'{split}2017')
        annotation_file = os.path.join(root, 'annotations', f'instances_{split}2017.json')
        
        if not os.path.exists(image_dir):
            raise ValueError(f"COCO image directory not found: {image_dir}")
        
        # Load annotations if available
        self.images = []
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            # Build image id to file name mapping
            id_to_file = {img['id']: img['file_name'] for img in annotations['images']}
            
            # Get images with annotations
            for ann in annotations['annotations']:
                img_id = ann['image_id']
                if img_id in id_to_file:
                    img_path = os.path.join(image_dir, id_to_file[img_id])
                    if os.path.exists(img_path):
                        self.images.append(img_path)
        else:
            # If no annotations, just use all images
            for img_file in os.listdir(image_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(image_dir, img_file))
        
        if max_samples and len(self.images) > max_samples:
            self.images = self.images[:max_samples]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Convert to [0, 255] range
        if isinstance(image, torch.Tensor):
            if image.max() <= 1.0:
                image = (image * 255).long().clamp(0, 255)
        else:
            image = transforms.ToTensor()(image)
            image = (image * 255).long().clamp(0, 255)
        
        # Return dummy label for COCO (0)
        return image, 0


def get_transform(dataset_name: str, image_size: int = 32, train: bool = True):
    """Get appropriate transforms for each dataset"""
    if dataset_name.lower() == 'cifar100':
        # CIFAR-100 uses 32x32 images
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.ToTensor()
    else:
        # ImageNet and COCO use larger images
        if train:
            transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])
    
    return transform


def get_dataloader(
    dataset_name: str,
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 32,
    train: bool = True,
    max_samples: int = None,
    shuffle: bool = True
):
    """
    Get DataLoader for specified dataset
    """
    transform = get_transform(dataset_name, image_size, train)
    
    if dataset_name.lower() == 'cifar100':
        # CIFAR100Dataset handles the path correctly (root is the parent directory)
        dataset = CIFAR100Dataset(root=root, train=train, transform=transform)
        num_classes = 100
    elif dataset_name.lower() == 'imagenet' or dataset_name.lower() == 'imagenet10k':
        dataset = ImageNet10kDataset(
            root=root,
            split='train' if train else 'val',
            transform=transform,
            max_samples=max_samples or 10000
        )
        num_classes = 1000  # ImageNet has 1000 classes
    elif dataset_name.lower() == 'coco':
        dataset = COCODataset(
            root=root,
            split='train' if train else 'val',
            transform=transform,
            max_samples=max_samples
        )
        num_classes = 80  # COCO has 80 object classes
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train
    )
    
    return dataloader, num_classes

