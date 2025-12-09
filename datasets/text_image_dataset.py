"""
Text-Image Dataset Loaders for LLM-based Prompt Image Generation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
from typing import Optional, List, Tuple
import random

from .dataset_loaders import CIFAR100Dataset, ImageNet10kDataset, COCODataset
from models.llm_text_encoder import SimpleTokenizer


class TextImageDataset(Dataset):
    """
    Base class for text-image pair datasets
    """
    def __init__(
        self,
        image_dataset: Dataset,
        texts: Optional[List[str]] = None,
        tokenizer=None,
        max_text_length: int = 512,
        generate_texts: bool = False,
        num_classes: Optional[int] = None
    ):
        self.image_dataset = image_dataset
        self.max_text_length = max_text_length
        
        if tokenizer is None:
            from models.llm_text_encoder import SimpleTokenizer
            self.tokenizer = SimpleTokenizer(max_length=max_text_length)
        else:
            self.tokenizer = tokenizer
        
        # Handle texts
        if texts is not None:
            self.texts = texts
        elif generate_texts:
            # Generate simple descriptive texts from labels
            self.texts = self._generate_texts_from_labels(num_classes)
        else:
            # Use generic texts
            self.texts = [f"An image of class {i % (num_classes or 100)}" 
                         for i in range(len(image_dataset))]
        
        # Ensure texts match dataset length
        if len(self.texts) != len(image_dataset):
            # Repeat or truncate
            if len(self.texts) < len(image_dataset):
                self.texts = self.texts * (len(image_dataset) // len(self.texts) + 1)
            self.texts = self.texts[:len(image_dataset)]
    
    def _generate_texts_from_labels(self, num_classes: Optional[int]) -> List[str]:
        """Generate simple descriptive texts from class labels"""
        generic_descriptions = [
            "a beautiful image", "a colorful picture", "a detailed photograph",
            "an interesting scene", "a clear image", "a nice picture"
        ]
        return [random.choice(generic_descriptions) for _ in range(len(self.image_dataset))]
    
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        text = self.texts[idx]
        
        # Tokenize text
        tokenized = self.tokenizer([text], padding=True, truncation=True)
        
        return {
            'image': image,
            'label': label,
            'text': text,
            'token_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0)
        }


class CIFAR100TextDataset(TextImageDataset):
    """CIFAR-100 with text descriptions"""
    CIFAR100_CLASSES = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]
    
    def __init__(self, root: str, train: bool = True, transform=None, tokenizer=None, max_text_length: int = 512):
        from .dataset_loaders import CIFAR100Dataset
        image_dataset = CIFAR100Dataset(root=root, train=train, transform=transform)
        
        # Generate descriptive texts from class names
        texts = []
        for idx in range(len(image_dataset)):
            _, label = image_dataset[idx]
            texts.append(f"a photo of a {self.CIFAR100_CLASSES[label]}")
        
        super().__init__(
            image_dataset=image_dataset,
            texts=texts,
            tokenizer=tokenizer,
            max_text_length=max_text_length,
            num_classes=100
        )


class ImageNetTextDataset(TextImageDataset):
    """ImageNet with text descriptions"""
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        max_samples: int = 10000,
        tokenizer=None,
        max_text_length: int = 512
    ):
        from .dataset_loaders import ImageNet10kDataset
        image_dataset = ImageNet10kDataset(
            root=root,
            split=split,
            transform=transform,
            max_samples=max_samples
        )
        
        # Generate texts from class names
        texts = []
        for idx in range(len(image_dataset)):
            _, label = image_dataset[idx]
            # Reverse lookup class name from class_to_idx
            class_name = f"class_{label}"
            for name, cls_idx in image_dataset.class_to_idx.items():
                if cls_idx == label:
                    class_name = name
                    break
            texts.append(f"a photo of a {class_name.replace('_', ' ')}")
        
        super().__init__(
            image_dataset=image_dataset,
            texts=texts,
            tokenizer=tokenizer,
            max_text_length=max_text_length,
            num_classes=1000
        )


class COCOTextDataset(TextImageDataset):
    """COCO dataset with captions"""
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        max_samples: int = None,
        tokenizer=None,
        max_text_length: int = 512
    ):
        from .dataset_loaders import COCODataset
        image_dataset = COCODataset(
            root=root,
            split=split,
            transform=transform,
            max_samples=max_samples
        )
        
        # Load COCO captions
        annotation_file = os.path.join(root, 'annotations', f'captions_{split}2017.json')
        texts = self._load_coco_captions(annotation_file, image_dataset)
        
        super().__init__(
            image_dataset=image_dataset,
            texts=texts,
            tokenizer=tokenizer,
            max_text_length=max_text_length,
            num_classes=80
        )
    
    def _load_coco_captions(self, annotation_file: str, image_dataset: Dataset) -> List[str]:
        """Load captions from COCO annotation file"""
        if not os.path.exists(annotation_file):
            print(f"Warning: COCO captions file not found: {annotation_file}")
            print("Using generic descriptions instead")
            return [f"an image from the COCO dataset" for _ in range(len(image_dataset))]
        
        # Build image filename to captions mapping
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        image_id_to_captions = {}
        for ann in annotations['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            if image_id not in image_id_to_captions:
                image_id_to_captions[image_id] = []
            image_id_to_captions[image_id].append(caption)
        
        # Match captions to dataset images
        texts = []
        for idx in range(len(image_dataset)):
            image_path = image_dataset.images[idx]
            image_filename = os.path.basename(image_path)
            
            # Find image ID from filename (COCO format: COCO_train2017_000000123456.jpg)
            try:
                image_id = int(image_filename.split('_')[-1].split('.')[0])
                if image_id in image_id_to_captions:
                    # Use first caption
                    texts.append(image_id_to_captions[image_id][0])
                else:
                    texts.append("an image from the COCO dataset")
            except:
                texts.append("an image from the COCO dataset")
        
        return texts


def get_text_image_dataloader(
    dataset_name: str,
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 32,
    train: bool = True,
    max_samples: int = None,
    tokenizer=None,
    max_text_length: int = 512,
    shuffle: bool = True
):
    """
    Get DataLoader for text-image pair datasets
    """
    from .dataset_loaders import get_transform
    transform = get_transform(dataset_name, image_size, train)
    
    if dataset_name.lower() == 'cifar100':
        dataset = CIFAR100TextDataset(
            root=root,
            train=train,
            transform=transform,
            tokenizer=tokenizer,
            max_text_length=max_text_length
        )
        num_classes = 100
    elif dataset_name.lower() == 'imagenet' or dataset_name.lower() == 'imagenet10k':
        dataset = ImageNetTextDataset(
            root=root,
            split='train' if train else 'val',
            transform=transform,
            max_samples=max_samples or 10000,
            tokenizer=tokenizer,
            max_text_length=max_text_length
        )
        num_classes = 1000
    elif dataset_name.lower() == 'coco':
        dataset = COCOTextDataset(
            root=root,
            split='train' if train else 'val',
            transform=transform,
            max_samples=max_samples,
            tokenizer=tokenizer,
            max_text_length=max_text_length
        )
        num_classes = 80
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def collate_fn(batch):
        """Custom collate function for text-image pairs"""
        images = torch.stack([item['image'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        token_ids = torch.stack([item['token_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'images': images,
            'labels': labels,
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'texts': [item['text'] for item in batch]
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        collate_fn=collate_fn
    )
    
    return dataloader, num_classes

