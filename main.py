"""
Car Detection and Classification System
Main entry point for the project

Usage:
    # Train detection model
    python main.py train --task detection --data-dir ./data/cars --epochs 100
    
    # Train classification model
    python main.py train --task classification --data-dir ./data/car_types --epochs 50
    
    # Run inference on video
    python main.py infer --input video.mp4 --output result.mp4
    
    # Evaluate model
    python main.py evaluate --task classification --model checkpoint.pth --data-dir ./data
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def train(args):
    """Train model"""
    from train import train_detector, train_classifier
    
    if args.task == 'detection':
        train_detector(args)
    else:
        train_classifier(args)


def infer(args):
    """Run inference"""
    from inference import CarRetrievalSystem
    
    system = CarRetrievalSystem(
        detector_path=args.detector,
        classifier_path=args.classifier,
        classifier_config={'architecture': args.classifier_arch},
        device=args.device,
        use_ultralytics=args.use_ultralytics
    )
    
    ext = Path(args.input).suffix.lower()
    
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        annotated, results = system.process_image(
            args.input, args.output, args.conf
        )
        print(f"\nDetected {len(results)} vehicles:")
        for r in results:
            print(f"  - {r['car_type']}: {r['type_conf']:.2f}")
    else:
        stats = system.process_video(
            args.input, args.output, args.conf, args.skip_frames
        )
        system.print_statistics(stats)
        print(f"\nOutput saved to {args.output}")


def evaluate(args):
    """Evaluate model"""
    import torch
    from torch.utils.data import DataLoader
    from utils.evaluation import ClassificationEvaluator, DetectionEvaluator
    from utils.dataset import CarClassificationDataset, CarDetectionDataset
    from models.classifier import get_classifier
    from models.detector import create_car_detector
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.task == 'classification':
        # Load model
        model = get_classifier(args.architecture, num_classes=7, pretrained=False)
        if args.model:
            checkpoint = torch.load(args.model, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Load data
        dataset = CarClassificationDataset(args.data_dir, 'val')
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Evaluate
        evaluator = ClassificationEvaluator(model, dataloader, device)
        evaluator.generate_report(args.output_dir)
    
    else:
        # Load model
        model = create_car_detector(args.model_size, num_classes=1)
        if args.model:
            checkpoint = torch.load(args.model, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Load data
        from utils.dataset import collate_fn
        dataset = CarDetectionDataset(args.data_dir, 'val')
        dataloader = DataLoader(
            dataset, batch_size=16, shuffle=False,
            num_workers=4, collate_fn=collate_fn
        )
        
        # Evaluate
        evaluator = DetectionEvaluator(model, dataloader, device)
        evaluator.generate_report(args.output_dir)


def demo(args):
    """Run demonstration"""
    print("Running demonstration...")
    print("This will download a sample video and run inference.")
    
    from inference import CarRetrievalSystem
    
    # Initialize system
    system = CarRetrievalSystem(
        use_ultralytics=True
    )
    
    # Process sample image (create a test image)
    import numpy as np
    import cv2
    
    # Create a simple test image
    test_img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (100, 100), (300, 250), (0, 255, 0), 2)
    
    test_path = './test-assets/test_image.jpg'
    output_path = './test-output/test_output.jpg'
    cv2.imwrite(test_path, test_img)
    
    # Process
    annotated, results = system.process_frame(test_img)
    cv2.imwrite(output_path, annotated)
    
    print(f"Demo complete! Output saved to {output_path}")
    print(f"Detections: {len(results)}")


def main():
    parser = argparse.ArgumentParser(
        description='Car Detection and Classification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--task', type=str, choices=['detection', 'classification'],
                              default='classification', help='Task to train')
    train_parser.add_argument('--data-dir', type=str, required=True, help='Dataset directory')
    train_parser.add_argument('--output-dir', type=str, default='./checkpoints', help='Output directory')
    train_parser.add_argument('--architecture', type=str, default='resnet50', help='Model architecture')
    train_parser.add_argument('--model-size', type=str, default='small', help='Detector size')
    train_parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--img-size', type=int, default=224, help='Image size')
    train_parser.add_argument('--workers', type=int, default=4, help='Data loader workers')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--input', type=str, required=True, help='Input video/image path')
    infer_parser.add_argument('--output', type=str, default='output.mp4', help='Output path')
    infer_parser.add_argument('--detector', type=str, default=None, help='Detector checkpoint')
    infer_parser.add_argument('--classifier', type=str, default=None, help='Classifier checkpoint')
    infer_parser.add_argument('--classifier-arch', type=str, default='resnet50', help='Classifier architecture')
    infer_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    infer_parser.add_argument('--skip-frames', type=int, default=0, help='Frames to skip')
    infer_parser.add_argument('--device', type=str, default='auto', help='Device')
    infer_parser.add_argument('--use-ultralytics', action='store_true', help='Use YOLOv8')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--task', type=str, choices=['detection', 'classification'],
                             default='classification', help='Task to evaluate')
    eval_parser.add_argument('--model', type=str, help='Model checkpoint')
    eval_parser.add_argument('--data-dir', type=str, required=True, help='Dataset directory')
    eval_parser.add_argument('--output-dir', type=str, default='./eval_results', help='Output directory')
    eval_parser.add_argument('--architecture', type=str, default='resnet50', help='Model architecture')
    eval_parser.add_argument('--model-size', type=str, default='small', help='Detector size')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'infer':
        infer(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'demo':
        demo(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
