# Car Detection and Classification System

A comprehensive deep learning system for detecting and classifying Indonesian car types in images and videos. Built with PyTorch, featuring custom-built neural network architectures including YOLO-based object detection and multiple classification models (CNN- and Transformer-based).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Model Architectures](#model-architectures)
  - [Object Detection](#object-detection)
  - [Classification](#classification)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project implements a complete deep learning pipeline for vehicle understanding tasks:

1. **Object Detection** – Locating multiple car instances in images or videos  
2. **Car Type Classification** – Identifying the type of each detected vehicle  

### Supported Car Types (Indonesian Market Focus)

| Type        | Examples |
|------------|----------|
| Sedan      | Toyota Vios, Honda City, Toyota Camry |
| SUV        | Toyota Fortuner, Mitsubishi Pajero, Honda CR-V |
| MPV        | Toyota Avanza, Mitsubishi Xpander, Honda Mobilio |
| Hatchback | Honda Jazz, Toyota Yaris, Suzuki Baleno |
| Pickup    | Toyota Hilux, Mitsubishi Triton, Isuzu D-Max |
| Minivan   | Toyota HiAce, Daihatsu Gran Max |
| Crossover | Honda HR-V, Toyota C-HR, Mazda CX-3 |

---

## Features

### Core Capabilities

- Multi-scale object detection for varied vehicle sizes  
- Seven-class car type classification  
- Image and video inference support  
- Batch processing for large datasets  

### Technical Highlights

- Custom neural network architectures (not simple fine-tuning)
- CNN- and Transformer-based classifiers
- Ensemble inference for accuracy improvement
- Mixed-precision training support
- Comprehensive evaluation metrics (mAP, F1-score, confusion matrix)
- Visualized outputs with annotated bounding boxes  

### Training Features

- Learning-rate warmup and cosine annealing  
- Early stopping  
- Gradient accumulation  
- Advanced data augmentation  
- Automatic checkpointing  

---

## Project Structure

```text
car-detection-system/
│
├── main.py                  # Unified CLI entry point
├── train.py                 # Training pipelines
├── inference.py             # Video and image inference
├── requirements.txt         # Dependencies
├── README.md                # Documentation
│
├── models/
│   ├── detector.py          # YOLO-based detector
│   └── classifier.py        # Classification models
│
├── utils/
│   ├── dataset.py           # Dataset and augmentation utilities
│   └── evaluation.py        # Metrics and evaluation logic
│
├── configs/
│   └── config.yaml
│
├── checkpoints/             # Saved models
├── data/                    # Datasets
├── outputs/                 # Inference results
└── notebooks/               # Experiments and analysis
```

---

## Installation

### Prerequisites

* Python 3.8 or higher
* CUDA 11.0+ (optional, for GPU acceleration)
* Minimum 8 GB RAM (16 GB recommended)

### Clone the Repository

```bash
git clone https://github.com/delfrinandopranata/car-detection-system.git
cd car-detection-system
```

### Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Option 1: Run Demo
```bash
python main.py demo
```

### Option 2: Process a Video (Using Pre-trained YOLOv8)
```bash
python main.py infer \
  --input ./test-assets/traffic_test.mp4 \
  --output ./test-output/traffic_test.mp4 \
    --use-ultralytics \
    --conf 0.25
```

### Option 3: Process Local Image
```bash
python main.py infer \
  --input ./test-assets/car-image.png \
  --output ./test-output/car-image.png \
    --use-ultralytics
```

---

## Usage

### Training

#### Train Classification Model

```bash
python main.py train \
  --task classification \
  --data-dir ./data/car_types \
  --architecture resnet50 \
  --pretrained \
  --epochs 50 \
  --batch-size 32
```

#### Train Detection Model

```bash
python main.py train \
  --task detection \
  --data-dir ./data/cars \
  --model-size small \
  --epochs 100 \
  --img-size 640
```

### Inference

Use `main.py infer` to process images or videos with trained models.

### Evaluation

Evaluation metrics include accuracy, F1-score, confusion matrix (classification), and mAP@0.5 (detection).

---

## Model Architectures

### Object Detection

* CSPDarknet backbone
* PANet feature pyramid
* Anchor-based multi-scale detection

### Classification

* Custom ResNet with SE-attention
* Vision Transformer (ViT)
* EfficientNet and Swin Transformer support
* Optional ensemble inference

---

## Dataset Preparation

### Classification Dataset Structure

```text
data/car_types/
├── train/
├── val/
└── test/
```

### Detection Dataset (YOLO Format)

```text
data/cars/
├── train/
│   ├── images/
│   └── labels/
├── val/
└── data.yaml
```

---

## Configuration

All configuration parameters are defined in `configs/config.yaml`, covering:

* Dataset paths
* Model selection
* Training hyperparameters
* Augmentation settings
* Inference thresholds

---

## API Reference

```python
from inference import CarRetrievalSystem

system = CarRetrievalSystem(use_ultralytics=True)
system.process_video("input.mp4", "output.mp4")
```

---

## Performance

### Classification

| Model           | Accuracy | F1-score |
| --------------- | -------- | -------- |
| ResNet-50       | 92.4%    | 0.92     |
| EfficientNet-B2 | 93.1%    | 0.93     |
| Ensemble        | 94.6%    | 0.94     |

### Detection

| Model Size | mAP@0.5 |
| ---------- | ------- |
| Small      | 52.8%   |
| Medium     | 58.4%   |
| Large      | 62.1%   |

---

## Troubleshooting

* Reduce batch size for GPU memory constraints
* Enable mixed-precision training
* Increase dataset diversity for improved accuracy

---

## Contact

**Delfrinando Pranata**  
Email: [delfrinando@gmail.com](mailto:delfrinando@gmail.com)  
Phone: +60172926313