"""
Integrated Inference Pipeline
Combines Car Detection and Car Type Classification for video processing
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.detector import create_car_detector, CarDetectionModel
from models.classifier import get_classifier, CAR_TYPES


# ============================================================================
# Integrated Pipeline
# ============================================================================

class CarRetrievalSystem:
    """
    Complete Car Retrieval System
    Combines object detection with car type classification
    """
    
    def __init__(
        self,
        detector_path: Optional[str] = None,
        classifier_path: Optional[str] = None,
        detector_config: Dict = None,
        classifier_config: Dict = None,
        device: str = 'auto',
        use_ultralytics: bool = True  # Use YOLOv8 from ultralytics for better performance
    ):
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.use_ultralytics = use_ultralytics
        
        # Initialize detector
        self._init_detector(detector_path, detector_config)
        
        # Initialize classifier
        self._init_classifier(classifier_path, classifier_config)
        
        # Car type labels
        self.car_types = CAR_TYPES
        
        # Color palette for visualization
        self.colors = {
            'sedan': (255, 100, 100),
            'suv': (100, 255, 100),
            'mpv': (100, 100, 255),
            'hatchback': (255, 255, 100),
            'pickup': (255, 100, 255),
            'minivan': (100, 255, 255),
            'crossover': (200, 200, 200)
        }
        
        # Statistics
        self.stats = defaultdict(int)
    
    def _init_detector(self, model_path: Optional[str], config: Dict):
        """Initialize object detector"""
        if self.use_ultralytics:
            try:
                from ultralytics import YOLO
                # Use pre-trained YOLOv8 for car detection
                if model_path and os.path.exists(model_path):
                    self.detector = YOLO(model_path)
                else:
                    # Use YOLOv8n pre-trained on COCO (includes 'car' class)
                    self.detector = YOLO('yolov8n.pt')
                print("Using Ultralytics YOLOv8 detector")
                return
            except ImportError:
                print("Ultralytics not available, using custom detector")
        
        # Use custom detector
        config = config or {'model_size': 'small', 'num_classes': 1}
        self.detector = create_car_detector(
            model_size=config.get('model_size', 'small'),
            num_classes=config.get('num_classes', 1)
        ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.detector.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded detector from {model_path}")
        
        self.detector.eval()
    
    def _init_classifier(self, model_path: Optional[str], config: Dict):
        """Initialize car type classifier"""
        config = config or {'architecture': 'resnet50', 'num_classes': 7}
        
        self.classifier = get_classifier(
            config.get('architecture', 'resnet50'),
            num_classes=config.get('num_classes', 7),
            pretrained=True
        ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded classifier from {model_path}")
        
        self.classifier.eval()
    
    def preprocess_image(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """Preprocess image for classifier"""
        # Resize
        image_resized = cv2.resize(image, target_size)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor.unsqueeze(0).to(self.device)
    
    def detect_cars(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict]:
        """
        Detect cars in image
        Returns list of detections with boxes and confidence
        """
        detections = []
        
        if self.use_ultralytics:
            # Use YOLOv8
            results = self.detector(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        # Class 2 is 'car' in COCO, class 5 is 'bus', class 7 is 'truck'
                        if cls_id in [2, 5, 7]:  # car, bus, truck
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            detections.append({
                                'box': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': conf,
                                'class': 'vehicle'
                            })
        else:
            # Use custom detector
            img_tensor = self.preprocess_image(image, (640, 640))
            with torch.no_grad():
                predictions = self.detector.predict(
                    img_tensor, conf_threshold, iou_threshold
                )
            
            h, w = image.shape[:2]
            scale_x, scale_y = w / 640, h / 640
            
            for pred in predictions:
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                
                for box, score in zip(boxes, scores):
                    x1 = int(box[0] * scale_x)
                    y1 = int(box[1] * scale_y)
                    x2 = int(box[2] * scale_x)
                    y2 = int(box[3] * scale_y)
                    
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': float(score),
                        'class': 'vehicle'
                    })
        
        return detections
    
    def classify_car_type(self, car_crop: np.ndarray) -> Tuple[str, float]:
        """
        Classify the type of car from cropped image
        Returns (car_type, confidence)
        """
        # Preprocess
        car_tensor = self.preprocess_image(car_crop)
        
        # Classify
        with torch.no_grad():
            outputs = self.classifier(car_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = probs.max(1)
        
        car_type = self.car_types[pred.item()]
        confidence = conf.item()
        
        return car_type, confidence
    
    def process_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        classify: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process single frame
        Returns annotated frame and list of detections with classifications
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect cars
        detections = self.detect_cars(rgb_frame, conf_threshold)
        
        results = []
        annotated_frame = frame.copy()
        
        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = box
            
            # Ensure valid box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            
            # Crop car region
            car_crop = rgb_frame[y1:y2, x1:x2]
            
            # Classify car type
            if classify and car_crop.size > 0:
                car_type, type_conf = self.classify_car_type(car_crop)
            else:
                car_type, type_conf = 'unknown', 0.0
            
            # Store result
            result = {
                'box': box,
                'detection_conf': det['confidence'],
                'car_type': car_type,
                'type_conf': type_conf
            }
            results.append(result)
            
            # Update statistics
            self.stats[car_type] += 1
            
            # Draw on frame
            color = self.colors.get(car_type, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{car_type}: {type_conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color, -1
            )
            cv2.putText(
                annotated_frame, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2
            )
        
        return annotated_frame, results
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        skip_frames: int = 0,
        show_progress: bool = True
    ) -> Dict:
        """
        Process video file
        Returns statistics dictionary
        """
        # Reset stats
        self.stats = defaultdict(int)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup output video
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        all_results = []
        
        pbar = tqdm(total=total_frames, desc='Processing video') if show_progress else None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames if needed
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                if writer:
                    writer.write(frame)
                if pbar:
                    pbar.update(1)
                continue
            
            # Process frame
            annotated_frame, results = self.process_frame(frame, conf_threshold)
            
            # Store results
            all_results.append({
                'frame': frame_count,
                'detections': results
            })
            
            # Write output
            if writer:
                writer.write(annotated_frame)
            
            if pbar:
                pbar.update(1)
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if pbar:
            pbar.close()
        
        # Compile statistics
        stats = {
            'total_frames': frame_count,
            'total_detections': sum(self.stats.values()),
            'car_type_counts': dict(self.stats),
            'detections_per_frame': all_results
        }
        
        return stats
    
    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        conf_threshold: float = 0.25
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Process single image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Process
        annotated, results = self.process_frame(image, conf_threshold)
        
        # Save output
        if output_path:
            cv2.imwrite(output_path, annotated)
        
        return annotated, results
    
    def print_statistics(self, stats: Dict):
        """Print detection statistics"""
        print("\n" + "=" * 50)
        print("Car Detection & Classification Results")
        print("=" * 50)
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Total vehicles detected: {stats['total_detections']}")
        print("\nCar Type Distribution:")
        print("-" * 30)
        
        for car_type, count in sorted(
            stats['car_type_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            percentage = count / max(stats['total_detections'], 1) * 100
            print(f"  {car_type:12s}: {count:5d} ({percentage:5.1f}%)")
        
        print("=" * 50)


# ============================================================================
# Video Download Utility
# ============================================================================

def download_video(url: str, output_path: str) -> str:
    """Download video from URL"""
    import urllib.request
    
    print(f"Downloading video from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")
    
    return output_path


# ============================================================================
# Main Inference Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Car Detection & Classification Inference')
    
    # Input
    parser.add_argument('--input', type=str, required=True,
                        help='Path to video or image file (or URL)')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Output path')
    
    # Model paths
    parser.add_argument('--detector', type=str, default=None,
                        help='Path to detector checkpoint')
    parser.add_argument('--classifier', type=str, default=None,
                        help='Path to classifier checkpoint')
    
    # Processing options
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='Number of frames to skip')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda)')
    
    # Architecture options
    parser.add_argument('--classifier-arch', type=str, default='resnet50',
                        help='Classifier architecture')
    parser.add_argument('--use-ultralytics', action='store_true',
                        help='Use Ultralytics YOLOv8')
    
    args = parser.parse_args()
    
    # Check if input is URL
    if args.input.startswith('http'):
        video_path = 'input_video.mp4'
        download_video(args.input, video_path)
    else:
        video_path = args.input
    
    # Initialize system
    system = CarRetrievalSystem(
        detector_path=args.detector,
        classifier_path=args.classifier,
        classifier_config={'architecture': args.classifier_arch},
        device=args.device,
        use_ultralytics=args.use_ultralytics
    )
    
    # Check if input is image or video
    ext = Path(video_path).suffix.lower()
    
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process image
        annotated, results = system.process_image(
            video_path, args.output, args.conf
        )
        print(f"Detected {len(results)} vehicles")
        for r in results:
            print(f"  - {r['car_type']}: {r['type_conf']:.2f}")
    else:
        # Process video
        stats = system.process_video(
            video_path,
            args.output,
            args.conf,
            args.skip_frames
        )
        system.print_statistics(stats)
        print(f"\nOutput saved to {args.output}")


if __name__ == '__main__':
    main()
