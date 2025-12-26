"""
Training Script for Car Detection and Classification Models
Supports training from scratch and transfer learning
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.detector import create_car_detector, CarDetectionModel
from models.classifier import (
    get_classifier, CustomResNet, VisionTransformer, 
    CAR_TYPES, CarTypeClassifier
)
from utils.dataset import (
    get_detection_dataloaders, get_classification_dataloaders,
    CarDetectionDataset, CarClassificationDataset
)


# ============================================================================
# Training Utilities
# ============================================================================

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int = 5
) -> optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler"""
    
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=epochs // 3, gamma=0.1
        )
    elif scheduler_type == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[epochs // 2, epochs * 3 // 4], gamma=0.1
        )
    else:
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    return scheduler


def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    """Linear warmup learning rate"""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# ============================================================================
# Detection Model Training
# ============================================================================

class DetectionTrainer:
    """Trainer for Car Detection Model"""
    
    def __init__(
        self,
        model: CarDetectionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        output_dir: str
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.get('learning_rate', 0.01),
            momentum=config.get('momentum', 0.937),
            weight_decay=config.get('weight_decay', 0.0005)
        )
        
        # Scheduler
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            config.get('scheduler', 'cosine'),
            config.get('epochs', 100)
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.get('amp', True) else None
        
        # Logging
        self.best_map = 0.0
        self.history = {'train_loss': [], 'val_loss': [], 'val_map': []}
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        loss_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for images, targets in pbar:
            images = images.to(self.device)
            
            # Convert targets to tensor format
            batch_targets = []
            for i, target in enumerate(targets):
                if target['boxes'].shape[0] > 0:
                    boxes = target['boxes']
                    labels = target['labels']
                    
                    # Convert to normalized center format
                    img_size = images.shape[-1]
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / img_size
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / img_size
                    w = (boxes[:, 2] - boxes[:, 0]) / img_size
                    h = (boxes[:, 3] - boxes[:, 1]) / img_size
                    
                    batch_idx = torch.full((boxes.shape[0], 1), i, device=self.device)
                    target_tensor = torch.cat([
                        batch_idx,
                        labels.unsqueeze(1).float().to(self.device),
                        cx.unsqueeze(1).to(self.device),
                        cy.unsqueeze(1).to(self.device),
                        w.unsqueeze(1).to(self.device),
                        h.unsqueeze(1).to(self.device)
                    ], dim=1)
                    batch_targets.append(target_tensor)
            
            if batch_targets:
                batch_targets = torch.cat(batch_targets, dim=0)
            else:
                batch_targets = torch.zeros((0, 6), device=self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    _, loss = self.model(images, batch_targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                _, loss = self.model(images, batch_targets)
                loss.backward()
                self.optimizer.step()
            
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix({'loss': loss_meter.avg})
        
        return loss_meter.avg
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        loss_meter = AverageMeter()
        
        all_predictions = []
        all_targets = []
        
        for images, targets in tqdm(self.val_loader, desc='Validating'):
            images = images.to(self.device)
            
            # Get predictions
            predictions = self.model.predict(images)
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            # Compute loss
            batch_targets = []
            for i, target in enumerate(targets):
                if target['boxes'].shape[0] > 0:
                    boxes = target['boxes']
                    labels = target['labels']
                    img_size = images.shape[-1]
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / img_size
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / img_size
                    w = (boxes[:, 2] - boxes[:, 0]) / img_size
                    h = (boxes[:, 3] - boxes[:, 1]) / img_size
                    
                    batch_idx = torch.full((boxes.shape[0], 1), i, device=self.device)
                    target_tensor = torch.cat([
                        batch_idx,
                        labels.unsqueeze(1).float().to(self.device),
                        cx.unsqueeze(1).to(self.device),
                        cy.unsqueeze(1).to(self.device),
                        w.unsqueeze(1).to(self.device),
                        h.unsqueeze(1).to(self.device)
                    ], dim=1)
                    batch_targets.append(target_tensor)
            
            if batch_targets:
                batch_targets = torch.cat(batch_targets, dim=0)
            else:
                batch_targets = torch.zeros((0, 6), device=self.device)
            
            _, loss = self.model(images.to(self.device), batch_targets)
            loss_meter.update(loss.item(), images.size(0))
        
        # Compute mAP (simplified)
        map_score = self._compute_map(all_predictions, all_targets)
        
        return loss_meter.avg, map_score
    
    def _compute_map(self, predictions: List[Dict], targets: List[Dict]) -> float:
        """Compute mean Average Precision (simplified version)"""
        # Simplified mAP calculation
        total_iou = 0
        total_matches = 0
        
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes']
            target_boxes = target['boxes']
            
            if len(pred_boxes) == 0 or len(target_boxes) == 0:
                continue
            
            # Compute IoU matrix
            for pb in pred_boxes:
                best_iou = 0
                for tb in target_boxes:
                    iou = self._compute_iou(pb.cpu().numpy(), tb.cpu().numpy())
                    best_iou = max(best_iou, iou)
                total_iou += best_iou
                total_matches += 1
        
        return total_iou / max(total_matches, 1)
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / max(union, 1e-6)
    
    def train(self, epochs: int):
        """Full training loop"""
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        early_stopping = EarlyStopping(patience=15)
        
        for epoch in range(1, epochs + 1):
            # Warmup
            warmup_lr(
                self.optimizer, epoch,
                self.config.get('warmup_epochs', 5),
                self.config.get('learning_rate', 0.01)
            )
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_map = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_map'].append(val_map)
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_mAP={val_map:.4f}")
            
            # Save best model
            if val_map > self.best_map:
                self.best_map = val_map
                self.save_checkpoint('best_detection.pth')
            
            # Early stopping
            early_stopping(val_map)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Save final model
        self.save_checkpoint('final_detection.pth')
        print(f"Training complete! Best mAP: {self.best_map:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_map': self.best_map,
            'history': self.history,
            'config': self.config
        }
        torch.save(checkpoint, self.output_dir / filename)


# ============================================================================
# Classification Model Training
# ============================================================================

class ClassificationTrainer:
    """Trainer for Car Type Classifier"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        output_dir: str
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            config.get('scheduler', 'cosine'),
            config.get('epochs', 50)
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.get('amp', True) else None
        
        # Logging
        self.best_acc = 0.0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            acc = (predicted == labels).float().mean()
            
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc.item(), images.size(0))
            
            pbar.set_postfix({'loss': loss_meter.avg, 'acc': acc_meter.avg})
        
        return loss_meter.avg, acc_meter.avg
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(self.val_loader, desc='Validating'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            acc = (predicted == labels).float().mean()
            
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc.item(), images.size(0))
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        return loss_meter.avg, acc_meter.avg
    
    def train(self, epochs: int):
        """Full training loop"""
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        early_stopping = EarlyStopping(patience=10)
        
        for epoch in range(1, epochs + 1):
            # Warmup
            warmup_lr(
                self.optimizer, epoch,
                self.config.get('warmup_epochs', 5),
                self.config.get('learning_rate', 0.001)
            )
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint('best_classifier.pth')
            
            # Early stopping
            early_stopping(val_acc)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Save final model
        self.save_checkpoint('final_classifier.pth')
        print(f"Training complete! Best accuracy: {self.best_acc:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'history': self.history,
            'config': self.config
        }
        torch.save(checkpoint, self.output_dir / filename)


# ============================================================================
# Main Training Functions
# ============================================================================

def train_detector(args):
    """Train car detection model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    config = {
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'weight_decay': 0.0005,
        'momentum': 0.937,
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'amp': True
    }
    
    # Create model
    model = create_car_detector(
        model_size=args.model_size,
        num_classes=1,  # Just 'car'
        pretrained=args.pretrained
    )
    
    # Create dataloaders
    train_loader, val_loader = get_detection_dataloaders(
        args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    
    # Create trainer
    trainer = DetectionTrainer(
        model, train_loader, val_loader,
        config, device, args.output_dir
    )
    
    # Train
    trainer.train(args.epochs)


def train_classifier(args):
    """Train car type classifier"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    config = {
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'weight_decay': 0.01,
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'amp': True
    }
    
    # Create model
    model = get_classifier(
        args.architecture,
        num_classes=7,  # Number of car types
        pretrained=args.pretrained
    )
    
    # Create dataloaders
    train_loader, val_loader = get_classification_dataloaders(
        args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    
    # Create trainer
    trainer = ClassificationTrainer(
        model, train_loader, val_loader,
        config, device, args.output_dir
    )
    
    # Train
    trainer.train(args.epochs)


def main():
    parser = argparse.ArgumentParser(description='Train Car Detection/Classification Models')
    
    # Task
    parser.add_argument('--task', type=str, choices=['detection', 'classification'],
                        default='detection', help='Task to train')
    
    # Data
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default='./checkpoints', help='Output directory')
    
    # Model
    parser.add_argument('--architecture', type=str, default='resnet50',
                        help='Model architecture (for classifier)')
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['nano', 'small', 'medium', 'large'],
                        help='Model size (for detector)')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=224, help='Input image size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    if args.task == 'detection':
        train_detector(args)
    else:
        train_classifier(args)


if __name__ == '__main__':
    main()
