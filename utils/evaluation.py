"""
Evaluation and Metrics for Car Detection and Classification
Includes mAP computation, confusion matrix, and detailed analysis
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.classifier import CAR_TYPES


# ============================================================================
# Object Detection Metrics
# ============================================================================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / max(union_area, 1e-6)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 11-point interpolation"""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 1
) -> Dict:
    """
    Compute mean Average Precision
    
    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels'
        ground_truths: List of dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes
    
    Returns:
        Dictionary with mAP and per-class AP
    """
    all_detections = []
    all_ground_truths = []
    
    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        # Predictions for this image
        if len(pred['boxes']) > 0:
            for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
                all_detections.append({
                    'image_id': img_idx,
                    'box': box,
                    'score': score,
                    'label': label,
                    'matched': False
                })
        
        # Ground truths for this image
        if len(gt['boxes']) > 0:
            for box, label in zip(gt['boxes'], gt['labels']):
                all_ground_truths.append({
                    'image_id': img_idx,
                    'box': box,
                    'label': label,
                    'matched': False
                })
    
    # Compute AP for each class
    aps = {}
    
    for cls_id in range(num_classes):
        # Get detections and GTs for this class
        cls_dets = [d for d in all_detections if d['label'] == cls_id]
        cls_gts = [g for g in all_ground_truths if g['label'] == cls_id]
        
        if len(cls_gts) == 0:
            aps[cls_id] = 0.0
            continue
        
        # Sort detections by score
        cls_dets.sort(key=lambda x: x['score'], reverse=True)
        
        # Match detections to ground truths
        tp = np.zeros(len(cls_dets))
        fp = np.zeros(len(cls_dets))
        
        for det_idx, det in enumerate(cls_dets):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(cls_gts):
                if gt['image_id'] != det['image_id'] or gt['matched']:
                    continue
                
                iou = compute_iou(det['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp[det_idx] = 1
                cls_gts[best_gt_idx]['matched'] = True
            else:
                fp[det_idx] = 1
        
        # Compute precision-recall curve
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        
        recalls = cumsum_tp / len(cls_gts)
        precisions = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-6)
        
        # Compute AP
        aps[cls_id] = compute_ap(recalls, precisions)
    
    # Compute mAP
    mAP = np.mean(list(aps.values()))
    
    return {
        'mAP': mAP,
        'AP_per_class': aps,
        'total_predictions': len(all_detections),
        'total_ground_truths': len(all_ground_truths)
    }


def compute_map_at_thresholds(
    predictions: List[Dict],
    ground_truths: List[Dict],
    thresholds: List[float] = None
) -> Dict:
    """Compute mAP at multiple IoU thresholds"""
    if thresholds is None:
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    maps = {}
    for thresh in thresholds:
        result = compute_map(predictions, ground_truths, thresh)
        maps[f'mAP@{thresh:.2f}'] = result['mAP']
    
    maps['mAP@[0.5:0.95]'] = np.mean(list(maps.values()))
    
    return maps


# ============================================================================
# Classification Metrics
# ============================================================================

class ClassificationEvaluator:
    """Evaluator for car type classification"""
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        class_names: List[str] = None
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.class_names = class_names or CAR_TYPES
        
        self.all_predictions = []
        self.all_labels = []
        self.all_probs = []
    
    def evaluate(self) -> Dict:
        """Run evaluation and compute metrics"""
        self.model.eval()
        
        with torch.no_grad():
            for images, labels in tqdm(self.dataloader, desc='Evaluating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                self.all_predictions.extend(preds.cpu().numpy())
                self.all_labels.extend(labels.cpu().numpy())
                self.all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy
        self.all_predictions = np.array(self.all_predictions)
        self.all_labels = np.array(self.all_labels)
        self.all_probs = np.array(self.all_probs)
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        return metrics
    
    def _compute_metrics(self) -> Dict:
        """Compute all classification metrics"""
        # Basic metrics
        accuracy = accuracy_score(self.all_labels, self.all_predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.all_labels, self.all_predictions, average=None
        )
        
        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            self.all_labels, self.all_predictions, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Classification report
        report = classification_report(
            self.all_labels, self.all_predictions,
            target_names=self.class_names[:len(np.unique(self.all_labels))],
            output_dict=True
        )
        
        metrics = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'per_class': {
                self.class_names[i]: {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i],
                    'support': int(support[i]),
                    'accuracy': per_class_accuracy[i]
                }
                for i in range(len(precision))
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        output_path: str = None,
        normalize: bool = True,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=self.class_names[:cm.shape[1]],
            yticklabels=self.class_names[:cm.shape[0]],
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Confusion matrix saved to {output_path}")
        
        plt.close()
    
    def plot_per_class_metrics(
        self,
        output_path: str = None,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """Plot per-class precision, recall, and F1"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.all_labels, self.all_predictions, average=None
        )
        
        x = np.arange(len(precision))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
        
        ax.set_xlabel('Car Type')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Classification Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names[:len(precision)], rotation=45)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Per-class metrics saved to {output_path}")
        
        plt.close()
    
    def generate_report(self, output_dir: str):
        """Generate full evaluation report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute metrics
        metrics = self.evaluate()
        
        # Save metrics to JSON
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(
            str(output_dir / 'confusion_matrix.png'),
            normalize=True
        )
        
        # Plot per-class metrics
        self.plot_per_class_metrics(
            str(output_dir / 'per_class_metrics.png')
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("Classification Evaluation Summary")
        print("=" * 60)
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1-Score:   {metrics['macro_f1']:.4f}")
        print(f"Weighted F1:      {metrics['weighted_f1']:.4f}")
        print("\nPer-Class Results:")
        print("-" * 50)
        
        for cls_name, cls_metrics in metrics['per_class'].items():
            print(f"  {cls_name:12s}: P={cls_metrics['precision']:.3f}, "
                  f"R={cls_metrics['recall']:.3f}, F1={cls_metrics['f1']:.3f}, "
                  f"N={cls_metrics['support']}")
        
        print("=" * 60)
        
        return metrics


# ============================================================================
# Detection Evaluator
# ============================================================================

class DetectionEvaluator:
    """Evaluator for car detection"""
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
    
    def evaluate(self) -> Dict:
        """Run evaluation"""
        self.model.eval()
        
        all_predictions = []
        all_ground_truths = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.dataloader, desc='Evaluating'):
                images = images.to(self.device)
                
                # Get predictions
                predictions = self.model.predict(images)
                
                for pred, target in zip(predictions, targets):
                    all_predictions.append({
                        'boxes': pred['boxes'].cpu().numpy(),
                        'scores': pred['scores'].cpu().numpy(),
                        'labels': pred['labels'].cpu().numpy()
                    })
                    all_ground_truths.append({
                        'boxes': target['boxes'].cpu().numpy(),
                        'labels': target['labels'].cpu().numpy()
                    })
        
        # Compute mAP
        metrics = compute_map(all_predictions, all_ground_truths)
        
        # Compute mAP at multiple thresholds
        map_thresholds = compute_map_at_thresholds(all_predictions, all_ground_truths)
        metrics.update(map_thresholds)
        
        return metrics
    
    def generate_report(self, output_dir: str) -> Dict:
        """Generate evaluation report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metrics = self.evaluate()
        
        # Save metrics
        with open(output_dir / 'detection_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Detection Evaluation Summary")
        print("=" * 60)
        print(f"mAP@0.50:      {metrics.get('mAP@0.50', metrics['mAP']):.4f}")
        print(f"mAP@[0.5:0.95]: {metrics.get('mAP@[0.5:0.95]', metrics['mAP']):.4f}")
        print(f"Total Predictions:   {metrics['total_predictions']}")
        print(f"Total Ground Truths: {metrics['total_ground_truths']}")
        print("=" * 60)
        
        return metrics


# ============================================================================
# Model Comparison
# ============================================================================

def compare_classifiers(
    models: Dict[str, nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str
) -> Dict:
    """Compare multiple classifier models"""
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        evaluator = ClassificationEvaluator(model, dataloader, device)
        metrics = evaluator.evaluate()
        results[name] = {
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1']
        }
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(results))
    width = 0.25
    
    accuracies = [r['accuracy'] for r in results.values()]
    macro_f1s = [r['macro_f1'] for r in results.values()]
    weighted_f1s = [r['weighted_f1'] for r in results.values()]
    
    ax.bar(x - width, accuracies, width, label='Accuracy')
    ax.bar(x, macro_f1s, width, label='Macro F1')
    ax.bar(x + width, weighted_f1s, width, label='Weighted F1')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(list(results.keys()), rotation=45)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=150)
    plt.close()
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':>12} {'Macro F1':>12} {'Weighted F1':>12}")
    print("-" * 70)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['accuracy']:>12.4f} "
              f"{metrics['macro_f1']:>12.4f} {metrics['weighted_f1']:>12.4f}")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    print("Evaluation module loaded successfully!")
    print("Available functions:")
    print("  - compute_map(): Compute mean Average Precision for detection")
    print("  - ClassificationEvaluator: Evaluate classification models")
    print("  - DetectionEvaluator: Evaluate detection models")
    print("  - compare_classifiers(): Compare multiple models")
