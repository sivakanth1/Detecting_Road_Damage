# Road Damage Detection using YOLOv7

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-ee4c2c)](https://pytorch.org/)
[![YOLOv7](https://img.shields.io/badge/YOLOv7-Custom-green)](https://github.com/WongKinYiu/yolov7)
[![Dataset](https://img.shields.io/badge/Dataset-RDD2022-orange)](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547)

A deep learning-based road damage detection system using YOLOv7, trained on the multi-national RDD2022 dataset. This project achieves **63.85% mAP@0.5** and **31.65% mAP@0.5:0.95** in detecting 4 types of road damage across images from 7 countries.

<!-- ![Road Damage Detection Demo](images/demo.png) -->

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Web Demo (Gradio)](#web-demo-gradio)
- [Results](#results)
- [References](#references)
- [License](#license)

---

## Overview

This project implements an automated road damage detection system using **YOLOv7** object detection architecture. The system can identify and classify 4 types of road damage:

| Class | Description | Example Count |
|-------|-------------|---------------|
| **D00** | Longitudinal Crack | 1,201 labels |
| **D10** | Transverse Crack | 611 labels |
| **D20** | Alligator Crack | 529 labels |
| **D40** | Pothole | 375 labels |

The model is trained on **21,983 images** from 7 countries and validated on **1,189 images** from the RDD2022 dataset.

---

## Features

- **Multi-country generalization** - Trained on diverse road conditions from China, Czech Republic, India, Japan, Norway, and United States
- **Ensemble inference** - Combines predictions from two models trained at different resolutions (640×640 and 512×512)
- **Interactive web interface** - Gradio-based UI for easy testing and demonstration
- **Custom YOLOv7 architecture** - Optimized for road damage detection with 37.2M parameters
- **Transfer learning** - Initialized from pre-trained YOLOv7 weights for faster convergence
- **Comprehensive logging** - TensorBoard integration for training monitoring

---

## Dataset

**Dataset:** [RDD2022 - The multi-national Road Damage Dataset](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547)

### Dataset Statistics

| Country | Training Images | Image Size | Notes |
|---------|----------------|------------|-------|
| China (Drone) | 482 | 512×512 | Aerial imagery |
| China (MotorBike) | 43 | 512×512 | Ground-level |
| Czech Republic | 1,757 | 600×600 | - |
| India | 4,483 | 720×720 | Largest subset |
| Japan | 2,606 | 600×600, 1080×1080 | Mixed resolutions |
| Norway | 5,247 | 3643-4040×2035-2044 | High-resolution |
| United States | 0 | - | Test set only |
| **Total** | **21,983** | - | **50,879 labels** |

---

## Model Performance

### Best Model: yolov7all6 (640×640 resolution)

**Training:** 100 epochs | **Batch Size:** 10 | **GPU:** NVIDIA RTX 4060 Laptop

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **63.85%** |
| **mAP@0.5:0.95** | **31.65%** |
| **Precision** | **68.67%** |
| **Recall** | **57.49%** |
| **Parameters** | 37.2M |
| **Training Time** | ~28 hours |

### Per-Class Performance

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| D00 | 66.3% | 63.5% | 66.6% | 36.0% |
| D10 | 68.6% | 53.2% | 61.3% | 29.5% |
| D20 | 71.7% | 61.2% | 69.6% | 36.7% |
| D40 | 68.1% | 52.0% | 58.0% | 24.4% |

---

## Installation

### Prerequisites

- Python 3.7 - 3.12
- CUDA 11.8+ (for GPU acceleration)
- Conda (recommended)

### Step 1: Clone YOLOv7 Repository

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
```

### Step 2: Replace Configuration Files

**IMPORTANT:** Replace the default YOLOv7 files with custom configurations from this project:

1. Copy `DetectingRoadDamage/folders_to_replace/training` → Replace `yolov7/cfg/training`
2. Copy `DetectingRoadDamage/folders_to_replace/data` → Replace `yolov7/data`


### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
```
torch>=1.7.0
torchvision>=0.8.1
opencv-python>=4.1.1
numpy>=1.18.5,<1.24.0
matplotlib>=3.2.2
tensorboard>=2.4.1
gradio==5.29.0
pillow>=7.1.2
tqdm
scipy
seaborn
```

### Step 4: Fix Environment Issues (Windows)

If you encounter `KMP_DUPLICATE_LIB_OK` errors on Windows:

```bash
# Option 1: Conda environment variable
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
conda deactivate
conda activate base  # or your environment name

# Option 2: Set system environment variable
set KMP_DUPLICATE_LIB_OK=TRUE
```

### Step 5: Download Dataset

1. Download RDD2022 dataset from [Figshare](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547)
2. Extract to your preferred location (e.g., `F:/RoadDamageDetection/Datasets/RDD2022/`)
3. Update paths in `data/crdd22.yaml`:

```yaml
train: /path/to/RDD2022/images/train
val: /path/to/RDD2022/images/val

nc: 4
names: ['D00', 'D10', 'D20', 'D40']
```

### Step 6: Download Pre-trained Weights

Download YOLOv7 pre-trained weights:

```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
```

---

## Dataset Preparation (REQUIRED Before Training)

**CRITICAL:** Before starting training, you MUST run these notebooks to prepare the dataset:

### Step 1: Validate Dataset Image Sizes

Run `CheckSizes.ipynb` to verify image dimensions and XML annotation consistency:

```bash
jupyter notebook CheckSizes.ipynb
```

**Purpose:**
- Validates image dimensions match XML annotations
- Identifies square vs. non-square images
- Detects resolution inconsistencies across countries
- Ensures dataset integrity before training

### Step 2: Convert Dataset to YOLOv7 Format

Run `Local_YOLOv7_DataConversion.ipynb` to convert RDD2022 annotations to YOLOv7 format:

```bash
jupyter notebook Local_YOLOv7_DataConversion.ipynb
```

**Purpose:**
- Converts XML annotations to YOLO txt format
- Normalizes bounding box coordinates
- Creates proper train/val splits
- Organizes images and labels in YOLOv7 structure

**Expected Output:**
```
RDD2022/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

**WARNING:** Skipping these steps will result in training failures or incorrect annotations!

---

## Project Structure

```
DetectingRoadDamage/
├── yolov7/                          # YOLOv7 implementation (~7.5GB - NOT included)
├── folders_to_replace/               # Trained model weights
│   ├── training                      # Custom model configurations [REPLACE yolov7/cfg/training]
│   └── data                          # Dataset configurations [REPLACE yolov7/data]
├── test_images/                           # Sample test images
├── index.ipynb                            # Ensemble inference + Gradio demo
├── CheckSizes.ipynb                       # [REQUIRED] Dataset validation utilities
├── Local_YOLOv7_DataConversion.ipynb      # [REQUIRED] Dataset format conversion
├── Directory_Structure_CRDDC_RDD2022.txt  # Dataset structure reference
├── commands to run.txt                    # Training command reference
├── yolov7all6-log.txt                     # Complete training logs
├── Documentation/
│   ├── PathFinders-Final Report.pdf
│   └── results_yolov7all6.txt
└── images/                                # Visualization results
```

---

## Usage

### Training

#### Model 1: yolov7all6 (640×640, 100 epochs)

```bash
cd yolov7

python train.py --workers 8 --device 0 --batch-size 10 \
  --data data/crdd22.yaml \
  --img 640 640 \
  --cfg cfg/training/yolov7-custom.yaml \
  --weights yolov7_training.pt \
  --name yolov7all6 \
  --hyp data/hyp.scratch.p5.custom.yaml \
  --epoch 100
```

#### Model 2: yolov7all61 (512×512, 200 epochs)

```bash
python train.py --workers 8 --device 0 --batch-size 10 \
  --data data/crdd22.yaml \
  --img 512 512 \
  --cfg cfg/training/yolov7-custom.yaml \
  --weights yolov7_training.pt \
  --name yolov7all61 \
  --hyp data/hyp.scratch.p5.custom.yaml \
  --epoch 200
```

#### Monitor Training with TensorBoard

```bash
cd yolov7
python -m tensorboard.main --logdir runs/train
```

Then open your browser to `http://localhost:6006`

#### Resume Training (if interrupted)

```bash
python train.py --resume
```

---

### Inference

#### Single Image/Folder Detection (Model 1)

```bash
python detect.py \
  --weights runs/train/yolov7all6/weights/best.pt \
  --conf 0.10 \
  --img 640 \
  --source /path/to/test_images \
  --save-conf
```

#### Single Image/Folder Detection (Model 2)

```bash
python detect.py \
  --weights runs/train/yolov7all61/weights/best.pt \
  --conf 0.25 \
  --img 512 \
  --source /path/to/test_images \
  --save-conf
```

**Parameters:**
- `--weights`: Path to trained model weights
- `--conf`: Confidence threshold (0.0-1.0)
- `--img`: Input image size
- `--source`: Path to images, videos, or folder
- `--save-conf`: Save confidence scores

---

### Web Demo (Gradio)

The project includes an interactive **Gradio web interface** for ensemble inference using both trained models.

#### Run the Demo

```bash
jupyter notebook index.ipynb
```

**Or run directly in Google Colab:**

1. Upload `index.ipynb` to Colab
2. Mount Google Drive or upload weights
3. Run all cells
4. Access the Gradio interface

#### Features:
- Upload single or multiple images
- Adjustable confidence threshold (0.1 - 0.9)
- Ensemble prediction from both models
- Visual bounding boxes with class labels
- Gallery view for batch results

#### Ensemble Pipeline

The demo combines predictions from:
1. **yolov7all6.pt** (640×640 resolution)
2. **yolov7all61.pt** (512×512 resolution)

Predictions are merged using **Non-Maximum Suppression (NMS)** for improved accuracy.

---

## Results

### Training Progression (yolov7all6)

| Epoch | Box Loss | Obj Loss | Cls Loss | mAP@0.5 | mAP@0.5:0.95 |
|-------|----------|----------|----------|---------|--------------|
| 0 | 0.0639 | 0.01289 | 0.01437 | 25.06% | 9.10% |
| 10 | 0.04682 | 0.0118 | 0.00574 | 46.34% | 19.94% |
| 50 | 0.04147 | 0.01102 | 0.00391 | 60.16% | 28.92% |
| 90 | 0.03702 | 0.01007 | 0.00250 | 63.25% | 31.20% |
| **99** | **0.03639** | **0.00985** | **0.00235** | **63.85%** | **31.65%** |

### Sample Detections

Place your sample detection images in the `images/` folder to showcase results.

---

## Key Files

| File | Description |
|------|-------------|
| `index.ipynb` | Ensemble inference pipeline + Gradio UI |
| `CheckSizes.ipynb` | **[REQUIRED]** Dataset validation and image size analysis |
| `Local_YOLOv7_DataConversion.ipynb` | **[REQUIRED]** Convert RDD2022 XML to YOLO format |
| `Directory_Structure_CRDDC_RDD2022.txt` | Expected dataset folder structure reference |
| `cfg/training/yolov7-custom.yaml` | Custom YOLOv7 architecture (4 classes) |
| `data/crdd22.yaml` | RDD2022 dataset configuration |
| `data/hyp.scratch.p5.custom.yaml` | Custom training hyperparameters |
| `yolov7all6-log.txt` | Complete training logs for Model 1 (100 epochs) |
| `yolov7all61-log.txt` | Complete training logs for Model 2 (200 epochs) |
| `commands to run.txt` | All training/inference commands used |

---

## Troubleshooting

### 1. CUDA Out of Memory
```bash
# Reduce batch size
python train.py --batch-size 4 ...  # Instead of 10
```

### 2. KMP Duplicate Library Error (Windows)
```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

### 3. Image Size Mismatch
Make sure to use the correct `--img` parameter:
- Model 1 (yolov7all6): `--img 640`
- Model 2 (yolov7all61): `--img 512`

### 4. Missing Weights
Download pre-trained weights:
```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
```

---

## References

### Dataset
```bibtex
@data{RDD2022,
  author = {Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and others},
  title = {RDD2022: The multi-national Road Damage Dataset},
  year = {2022},
  publisher = {figshare},
  doi = {10.6084/m9.figshare.21431547}
}
```

### YOLOv7
```bibtex
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

### Original Project
- **Repository:** [mdptlab/roaddamagedetector2022](https://github.com/mdptlab/roaddamagedetector2022)
- **YOLOv7 Implementation:** [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

---

## License

This project is based on YOLOv7 (GPL-3.0 License) and the RDD2022 dataset. Please refer to the original repositories for licensing information.

---

## Contributors

**Team:** PathFinders  
**Institution:** Texas A&M University - Corpus Christi  
**Course:** Capstone Project

---

## Acknowledgments

- YOLOv7 team for the excellent object detection framework
- RDD2022 dataset creators for the comprehensive road damage dataset
- Original roaddamagedetector2022 project for inspiration and guidance

---

**Last Updated:** November 2025
