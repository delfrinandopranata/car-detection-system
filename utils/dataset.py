"""
Dataset utilities for Car Detection and Classification
Includes data loading, augmentation, and preprocessing
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2


# ============================================================================
# Data Augmentation
# ============================================================================

class Compose:
    """Compose multiple transforms"""
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert image to tensor"""
    def __call__(self, image, target=None):
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, target


class Normalize:
    """Normalize image with ImageNet stats"""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, image, target=None):
        image = (image - self.mean) / self.std
        return image, target


class Resize:
    """Resize image and boxes"""
    def __init__(self, size: int):
        self.size = size
    
    def __call__(self, image, target=None):
        h, w = image.shape[:2]
        image = cv2.resize(image, (self.size, self.size))
        
        if target is not None and 'boxes' in target:
            boxes = target['boxes']
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * self.size / w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * self.size / h
            target['boxes'] = boxes
        
        return image, target


class RandomHorizontalFlip:
    """Random horizontal flip"""
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image, target=None):
        if random.random() < self.p:
            image = np.fliplr(image).copy()
            
            if target is not None and 'boxes' in target:
                w = image.shape[1]
                boxes = target['boxes']
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = boxes
        
        return image, target


class RandomColorJitter:
    """Random color augmentation"""
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, image, target=None):
        # Convert to HSV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Hue
        image[:, :, 0] += random.uniform(-self.hue, self.hue) * 180
        image[:, :, 0] = np.clip(image[:, :, 0], 0, 180)
        
        # Saturation
        image[:, :, 1] *= random.uniform(1 - self.saturation, 1 + self.saturation)
        image[:, :, 1] = np.clip(image[:, :, 1], 0, 255)
        
        # Value (brightness)
        image[:, :, 2] *= random.uniform(1 - self.brightness, 1 + self.brightness)
        image[:, :, 2] = np.clip(image[:, :, 2], 0, 255)
        
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return image, target


class RandomScale:
    """Random scaling"""
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2)):
        self.scale_range = scale_range
    
    def __call__(self, image, target=None):
        scale = random.uniform(*self.scale_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        image = cv2.resize(image, (new_w, new_h))
        
        if target is not None and 'boxes' in target:
            target['boxes'] = target['boxes'] * scale
        
        return image, target


class Mosaic:
    """Mosaic augmentation - combines 4 images"""
    def __init__(self, dataset, size: int = 640):
        self.dataset = dataset
        self.size = size
    
    def __call__(self, index: int) -> Tuple[np.ndarray, Dict]:
        indices = [index] + [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        
        mosaic_img = np.zeros((self.size * 2, self.size * 2, 3), dtype=np.uint8)
        all_boxes = []
        all_labels = []
        
        for i, idx in enumerate(indices):
            img, target = self.dataset.load_item(idx)
            h, w = img.shape[:2]
            
            # Position in mosaic
            if i == 0:  # top left
                x1, y1, x2, y2 = 0, 0, w, h
                dx, dy = 0, 0
            elif i == 1:  # top right
                x1, y1, x2, y2 = self.size, 0, self.size + w, h
                dx, dy = self.size, 0
            elif i == 2:  # bottom left
                x1, y1, x2, y2 = 0, self.size, w, self.size + h
                dx, dy = 0, self.size
            else:  # bottom right
                x1, y1, x2, y2 = self.size, self.size, self.size + w, self.size + h
                dx, dy = self.size, self.size
            
            # Clip to mosaic size
            x2 = min(x2, self.size * 2)
            y2 = min(y2, self.size * 2)
            
            mosaic_img[y1:y2, x1:x2] = img[:y2-y1, :x2-x1]
            
            if target is not None and 'boxes' in target:
                boxes = target['boxes'].copy()
                boxes[:, [0, 2]] += dx
                boxes[:, [1, 3]] += dy
                all_boxes.append(boxes)
                all_labels.append(target['labels'])
        
        # Resize mosaic
        mosaic_img = cv2.resize(mosaic_img, (self.size, self.size))
        scale = 0.5
        
        combined_target = {}
        if all_boxes:
            all_boxes = np.concatenate(all_boxes) * scale
            all_labels = np.concatenate(all_labels)
            
            # Clip boxes
            all_boxes[:, [0, 2]] = np.clip(all_boxes[:, [0, 2]], 0, self.size)
            all_boxes[:, [1, 3]] = np.clip(all_boxes[:, [1, 3]], 0, self.size)
            
            combined_target['boxes'] = all_boxes
            combined_target['labels'] = all_labels
        
        return mosaic_img, combined_target


# ============================================================================
# Datasets
# ============================================================================

class CarDetectionDataset(Dataset):
    """
    Dataset for car detection
    Supports YOLO and COCO format annotations
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        img_size: int = 640,
        transform: Optional[Callable] = None,
        annotation_format: str = 'yolo'  # 'yolo' or 'coco'
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.annotation_format = annotation_format
        
        self.images_dir = self.root_dir / 'images' / split
        self.labels_dir = self.root_dir / 'labels' / split
        
        # Get image list
        self.image_files = []
        if self.images_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                self.image_files.extend(list(self.images_dir.glob(ext)))
        
        # Default transform
        if self.transform is None:
            if split == 'train':
                self.transform = Compose([
                    Resize(img_size),
                    RandomHorizontalFlip(0.5),
                    RandomColorJitter(),
                    ToTensor(),
                    Normalize()
                ])
            else:
                self.transform = Compose([
                    Resize(img_size),
                    ToTensor(),
                    Normalize()
                ])
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def load_item(self, index: int) -> Tuple[np.ndarray, Dict]:
        """Load image and annotations without transforms"""
        img_path = self.image_files[index]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Load annotations
        if self.annotation_format == 'yolo':
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            target = self._load_yolo_annotations(label_path, w, h)
        else:
            target = self._load_coco_annotations(index, w, h)
        
        return image, target
    
    def _load_yolo_annotations(
        self,
        label_path: Path,
        img_w: int,
        img_h: int
    ) -> Dict:
        """Load YOLO format annotations"""
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        
                        # Convert to x1y1x2y2
                        x1 = (cx - bw / 2) * img_w
                        y1 = (cy - bh / 2) * img_h
                        x2 = (cx + bw / 2) * img_w
                        y2 = (cy + bh / 2) * img_h
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls)
        
        return {
            'boxes': np.array(boxes, dtype=np.float32).reshape(-1, 4),
            'labels': np.array(labels, dtype=np.int64)
        }
    
    def _load_coco_annotations(self, index: int, img_w: int, img_h: int) -> Dict:
        """Load COCO format annotations"""
        # Placeholder for COCO loading
        return {'boxes': np.array([]), 'labels': np.array([])}
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        image, target = self.load_item(index)
        
        if self.transform:
            image, target = self.transform(image, target)
        
        if target['boxes'].shape[0] > 0:
            target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros(0, dtype=torch.int64)
        
        return image, target


class CarClassificationDataset(Dataset):
    """
    Dataset for car type classification
    Expects folder structure: root/class_name/images
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        img_size: int = 224,
        transform: Optional[Callable] = None
    ):
        self.root_dir = Path(root_dir) / split
        self.img_size = img_size
        self.split = split
        
        # Car type classes
        self.classes = [
            'sedan', 'suv', 'mpv', 'hatchback',
            'pickup', 'minivan', 'crossover'
        ]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Get all images
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        # Set transform
        self.transform = transform
        if self.transform is None:
            if split == 'train':
                self.transform = self._get_train_transform()
            else:
                self.transform = self._get_val_transform()
    
    def _get_train_transform(self):
        def transform(image):
            # Resize
            image = cv2.resize(image, (self.img_size, self.img_size))
            
            # Random horizontal flip
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
            
            # Color jitter
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            image[:, :, 1] *= random.uniform(0.8, 1.2)
            image[:, :, 2] *= random.uniform(0.8, 1.2)
            image = np.clip(image, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
            
            # To tensor and normalize
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std
            
            return image
        return transform
    
    def _get_val_transform(self):
        def transform(image):
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std
            return image
        return transform
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================================
# Data Loaders
# ============================================================================

def collate_fn(batch):
    """Custom collate function for detection dataset"""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images)
    return images, targets


def get_detection_dataloaders(
    data_dir: str,
    img_size: int = 640,
    batch_size: int = 16,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Get train and validation dataloaders for detection"""
    
    train_dataset = CarDetectionDataset(data_dir, 'train', img_size)
    val_dataset = CarDetectionDataset(data_dir, 'val', img_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_classification_dataloaders(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Get train and validation dataloaders for classification"""
    
    train_dataset = CarClassificationDataset(data_dir, 'train', img_size)
    val_dataset = CarClassificationDataset(data_dir, 'val', img_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ============================================================================
# Dataset Preparation Utilities
# ============================================================================

def download_stanford_cars(output_dir: str):
    """Download Stanford Cars dataset"""
    import urllib.request
    import tarfile
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for Stanford Cars dataset
    urls = {
        'train_images': 'http://ai.stanford.edu/~jkrause/car196/cars_train.tgz',
        'test_images': 'http://ai.stanford.edu/~jkrause/car196/cars_test.tgz',
        'annotations': 'http://ai.stanford.edu/~jkrause/car196/cars_annos.mat'
    }
    
    for name, url in urls.items():
        print(f"Downloading {name}...")
        filename = output_dir / url.split('/')[-1]
        urllib.request.urlretrieve(url, filename)
        
        if filename.suffix == '.tgz':
            print(f"Extracting {name}...")
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall(output_dir)


def prepare_car_type_dataset(
    source_dir: str,
    output_dir: str,
    val_split: float = 0.2
):
    """
    Organize images into car type folders for classification
    
    Expected source structure:
    - source_dir/
        - images/
        - annotations.json  # with car type labels
    
    Output structure:
    - output_dir/
        - train/
            - sedan/
            - suv/
            - ...
        - val/
            - sedan/
            - suv/
            - ...
    """
    import shutil
    
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    car_types = ['sedan', 'suv', 'mpv', 'hatchback', 'pickup', 'minivan', 'crossover']
    
    # Create directories
    for split in ['train', 'val']:
        for car_type in car_types:
            (output_dir / split / car_type).mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    annotations_file = source_dir / 'annotations.json'
    if annotations_file.exists():
        with open(annotations_file) as f:
            annotations = json.load(f)
        
        # Shuffle and split
        random.shuffle(annotations)
        split_idx = int(len(annotations) * (1 - val_split))
        
        for i, ann in enumerate(annotations):
            img_name = ann['image']
            car_type = ann['type'].lower()
            
            if car_type not in car_types:
                continue
            
            split = 'train' if i < split_idx else 'val'
            src = source_dir / 'images' / img_name
            dst = output_dir / split / car_type / img_name
            
            if src.exists():
                shutil.copy(src, dst)
    
    print(f"Dataset prepared at {output_dir}")


def create_sample_dataset(output_dir: str, num_samples: int = 100):
    """Create a sample dataset with synthetic data for testing"""
    output_dir = Path(output_dir)
    
    car_types = ['sedan', 'suv', 'mpv', 'hatchback', 'pickup', 'minivan', 'crossover']
    
    # Create directories
    for split in ['train', 'val']:
        for car_type in car_types:
            (output_dir / split / car_type).mkdir(parents=True, exist_ok=True)
    
    # Create synthetic images
    for split in ['train', 'val']:
        n = num_samples if split == 'train' else num_samples // 5
        for car_type in car_types:
            for i in range(n):
                # Create random colored image
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img_path = output_dir / split / car_type / f"{car_type}_{i}.jpg"
                cv2.imwrite(str(img_path), img)
    
    print(f"Sample dataset created at {output_dir}")


if __name__ == '__main__':
    # Test dataset creation
    print("Testing dataset utilities...")
    
    # Create sample dataset
    create_sample_dataset('./data/sample_cars', num_samples=10)
    
    # Test classification dataset
    dataset = CarClassificationDataset('./data/sample_cars', 'train')
    print(f"Classification dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"Image shape: {img.shape}, Label: {label}")
    
    print("Dataset utilities test complete!")
