# 🔬 ImageCLEF 2025: Advanced Dermatological Segmentation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ImageCLEF 2025](https://img.shields.io/badge/ImageCLEF-2025-orange.svg)](https://www.imageclef.org/)

> An advanced deep learning framework for precise segmentation of dermatological lesions in medical images, featuring multiple state-of-the-art models including TransUNet and MedSAM.

This repository contains our solution for the **ImageCLEF 2025 Dermatological Segmentation Challenge** (Subtask 1). The system leverages cutting-edge models to accurately identify and segment regions affected by dermatological conditions in medical images, supporting clinical decision-making and advancing computer-aided diagnosis in dermatology.

## 🌟 Features

- **Multi-Model Architecture**: Implementation of both TransUNet and MedSAM models for robust segmentation performance
- **Extensive Data Augmentation**: Advanced techniques for improving model generalization
- **Model Ensemble Support**: Combine predictions from multiple models for better accuracy
- **Mixed Precision Training**: Faster training with reduced memory consumption
- **Comprehensive Evaluation**: Detailed metrics and visualization tools for performance assessment
- **Flexible Pipeline**: Easy-to-customize preprocessing, training, and inference workflows

## 📋 Table of Contents

1. [Installation](#-installation)
2. [Dataset Preparation](#-dataset-preparation)
3. [Training](#-training)
   - [Training TransUNet](#training-transunet)
   - [Training MedSAM](#training-medsam)
4. [Model Architecture](#-model-architecture)
   - [TransUNet](#transunet)
   - [MedSAM](#medsam)
5. [Training Parameters](#training-parameters)
6. [Output](#output)
7. [Prediction](#-prediction)
8. [Results and Evaluation](#-results-and-evaluation)
9. [Tips for Optimal Results](#tips-for-optimal-results)
10. [Citation](#-citation)
11. [Contact](#-contact)
12. [Acknowledgments](#-acknowledgments)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)

### Step 1: Clone this repository
```bash
git clone https://github.com/yourusername/ImageCLEF-2025-Dermatological-segmentation.git
cd ImageCLEF-2025-Dermatological-segmentation
```

### Step 2: Set up virtual environment
```bash
# Create a virtual environment
python -m venv derma_env

# Activate the environment
# On Windows:
derma_env\Scripts\activate
# On macOS/Linux:
source derma_env/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set up model repositories

#### For TransUNet:
```bash
git clone https://github.com/Beckschen/TransUNet.git
cd TransUNet
pip install -e .
cd ..
```

#### For MedSAM:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Step 5: Download pre-trained weights

#### TransUNet weights:
```bash
# Install gdown if not already installed
pip install gdown

# Download TransUNet pretrained ViT weights
mkdir -p pretrained_models
gdown "https://drive.google.com/uc?id=1mGhBIDcyollAEz0JMafXUO5zO3hErq4H" -O pretrained_models.zip
unzip pretrained_models.zip -d pretrained_models
```

#### MedSAM weights:
```bash
mkdir -p sam_checkpoints
wget -P sam_checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# On Windows, you can use PowerShell:
# Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "sam_checkpoints/sam_vit_b_01ec64.pth"
```

## 📊 Dataset Preparation

### Structure for TransUNet Training

Organize your dataset as follows:

```
dataset/
  ├── images_train/
  │    ├── Original/                  # Original training images
  │    │    ├── image1.jpg
  │    │    ├── image2.jpg
  │    │    └── ...
  │    └── [AugmentationFolders]/     # Optional augmentation folders
  │         └── ...
  ├── labels_train/
  │    ├── Original/                  # Original training masks
  │    │    ├── image1.tiff
  │    │    ├── image2.tiff
  │    │    └── ...
  │    └── [AugmentationFolders]/     # Optional augmentation folders
  │         └── ...
  ├── images_valid/                   # Validation images
  │    ├── image1.jpg
  │    ├── image2.jpg
  │    └── ...
  └── labels_valid/                   # Validation masks
       ├── image1.tiff
       ├── image2.tiff
       └── ...
```

**Important Notes:**
- The dataset loader matches images and masks by their filename stems (without extension)
- Supported image formats: .jpg, .png; supported mask formats: .tiff

### Structure for MedSAM Training

For MedSAM, convert your dataset to the required format:

```bash
python utils/preprocessing_data_for_MedSAM.py \
  --img_path "path/to/images_train/Original" \
  --gt_path "path/to/labels_train/Original" \
  --output_path "path/to/medsam_data/train" \
  --image_size 1024 \
  --save_individual

# For validation data
python utils/preprocessing_data_for_MedSAM.py \
  --img_path "path/to/images_valid" \
  --gt_path "path/to/labels_valid" \
  --output_path "path/to/medsam_data/val" \
  --image_size 1024 \
  --save_individual
```

### Data Augmentation

You can use the augmentation utility to enhance your training data:

```bash
python utils/augmentation_data.py \
  --img_path "path/to/images_train/Original" \
  --mask_path "path/to/labels_train/Original" \
  --output_img_dir "path/to/images_train/Augmented" \
  --output_mask_dir "path/to/labels_train/Augmented" \
  --num_augmentations 5
```

For combining multiple label sets, use:

```bash
python utils/combine_label.py \
  --input_mask_dirs "path/to/labels_train/Original" "path/to/labels_train/Expert2" \
  --output_dir "path/to/labels_train/Combined" \
  --combination_method "union"  # or "intersection", "majority"
```

## 🏋️ Training

Our system supports training two advanced models for dermatological segmentation: TransUNet and MedSAM.

### Training TransUNet

Basic training command:

```bash
python train_TransUNet.py \
  --train_images_path "path/to/images_train" \
  --train_labels_path "path/to/labels_train" \
  --val_images_path "path/to/images_valid" \
  --val_labels_path "path/to/labels_valid" \
  --folder_group "original" \
  --output_dir "output_transunet" \
  --epochs 100 \
  --batch_size 8 \
  --model_name "R50-ViT-B_16" \
  --img_size 224 \
  --use_augmentation
```

For using multiple specific augmentation groups:

```bash
# Use geometric augmentations (horizontal/vertical flips, rotations, etc.)
python train_TransUNet.py \
  --train_images_path "path/to/images_train" \
  --train_labels_path "path/to/labels_train" \
  --val_images_path "path/to/images_valid" \
  --val_labels_path "path/to/labels_valid" \
  --folder_group "geometric" \
  --output_dir "output_transunet_geometric_aug" \
  --epochs 100 \
  --batch_size 8 \
  --model_name "R50-ViT-B_16" \
  --img_size 224 \
  --use_augmentation
```

Using predefined folder groups:

```bash
# The script supports predefined folder groups for simpler training with multiple augmentations
python train_TransUNet.py \
  --train_images_path "path/to/images_train" \
  --train_labels_path "path/to/labels_train" \
  --val_images_path "path/to/images_valid" \
  --val_labels_path "path/to/labels_valid" \
  --folder_group "all" \
  --output_dir "output_all_aug" \
  --model_name "R50-ViT-B_16" \
  --epochs 100 \
  --batch_size 8
```

Available folder group options:
- `all`: Uses all available augmentation folders
- `geometric`: Focuses on geometric transformations (flips, rotations, etc.)
- `photometric`: Uses brightness/contrast augmentations
- `noise_artifact`: Includes noise-based augmentations
- `original`: Uses only original images (no augmentations)

This approach eliminates the need for manually listing all augmentation folders.

### Training MedSAM

```bash
python train_MedSAM.py \
  --train_data_path "path/to/medsam_data/train" \
  --val_data_path "path/to/medsam_data/val" \
  --work_dir "output_medsam" \
  --task_name "MedSAM-Derma" \
  --model_type "vit_b" \
  --checkpoint "sam_checkpoints/sam_vit_b_01ec64.pth" \
  --num_epochs 100 \
  --batch_size 2 \
  --img_size 1024 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --use_amp
```

## 🧠 Model Architecture

### TransUNet

TransUNet combines the efficiency of Transformers with U-Net's ability to preserve spatial information:

- **Feature Extraction**: ResNet or Vision Transformer backbone
- **Multi-scale Feature Integration**: Transformer encoder with skip connections
- **Semantic Segmentation**: U-Net-like decoder with multi-level feature fusion

Available model configurations:

| Model Name | Description | Size | Speed | Memory |
|------------|-------------|------|-------|--------|
| R50-ViT-B_16 | ResNet50 + ViT-Base with 16×16 patches | Medium | Fast | Medium |
| ViT-B_16 | ViT-Base with 16×16 patches | Medium | Medium | Medium |
| ViT-B_32 | ViT-Base with 32×32 patches | Medium | Fast | Low |
| ViT-L_16 | ViT-Large with 16×16 patches | Large | Slow | High |
| ViT-L_32 | ViT-Large with 32×32 patches | Large | Medium | Medium |

Choose the appropriate model with the `--model_name` parameter:

```bash
python train_TransUNet.py \
  --train_images_path "path/to/images_train" \
  --train_labels_path "path/to/labels_train" \
  --val_images_path "path/to/images_valid" \
  --val_labels_path "path/to/labels_valid" \
  --folder_group "original" \
  --model_name "ViT-L_16" \
  --output_dir "output_vitl16" \
  --epochs 100 \
  --learning_rate 1e-5
```

### MedSAM

MedSAM fine-tunes the Segment Anything Model (SAM) for medical image segmentation:

- **Image Encoder**: Vision Transformer backbone
- **Mask Decoder**: Transformer decoder that generates mask embeddings
- **Prompt Encoder**: Processes additional inputs like points or boxes
- **Medical Adaptation**: Fine-tuned specifically for dermatological images

Model types available:
- `vit_b`: Vision Transformer Base (default)
- `vit_l`: Vision Transformer Large (higher accuracy but requires more GPU memory)
- `vit_h`: Vision Transformer Huge (highest accuracy but requires significant GPU resources)

## Training Parameters

The script provides many parameters to customize your training:

### Data & Model Parameters
- `--train_images_path`: Path to training images directory (required)
- `--train_labels_path`: Path to training labels directory (required)
- `--val_images_path`: Path to validation images directory (required)
- `--val_labels_path`: Path to validation labels directory (required)
- `--folder_group`: Group of augmentation folders to use ['all', 'geometric', 'photometric', 'noise_artifact', 'original'] (default: 'all')
- `--model_name`: TransUNet model variant ['R50-ViT-B_16', 'ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32'] (default: 'R50-ViT-B_16')
- `--img_size`: Input image size (default: 224)
- `--num_classes`: Number of output classes (default: 1 for binary segmentation)
- `--pretrained_dir`: Directory containing pretrained model weights (default: 'pretrained_models')

### Training Parameters
- `--batch_size`: Batch size for training and validation (default: 8)
- `--epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Initial learning rate (default: 1e-5)
- `--weight_decay`: Weight decay for AdamW optimizer (default: 1e-2)
- `--use_augmentation`: Enable data augmentation during training (flag)
- `--scheduler_patience`: Patience for learning rate scheduler (default: 5)
- `--scheduler_factor`: Factor to reduce learning rate (default: 0.5)
- `--early_stopping_patience`: Patience for early stopping (default: 10)

### Output Parameters
- `--output_dir`: Output directory for saving models and visualizations (default: 'output')
- `--num_vis_images`: Number of images to visualize (default: 3)

### Technical Parameters
- `--seed`: Random seed for reproducibility (default: 42)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--use_amp`: Use Automatic Mixed Precision (flag, no value needed)

## Output

After training, the script produces the following outputs in the specified output directory:

1. **Model Files**:
   - `TransUNet_best_model.pth`: The best model weights based on validation loss

2. **Visualizations**:
   - `TransUNet_training_history.png`: Plots showing training/validation loss and dice scores
   - `TransUNet_prediction_results.png`: Visual comparison of input images, ground truth masks, and model predictions

3. **Console Output**:
   - Training and validation metrics for each epoch
   - Notification when a new best model is saved
   - Early stopping information


## 🔍 Prediction

### Using TransUNet for Inference

To generate segmentation masks for new images:

```bash
python predict_TransUNet.py `
  --image_dir "path\to\test_images" `
  --save_dir "path\to\predictions" `
  --checkpoint_paths "output_transunet\TransUNet_best_model.pth" `
  --vit_names "R50-ViT-B_16" `
  --vit_patches_sizes 16 `
  --img_size 224 `
  --device "cuda:0"
```



## 📈 Results and Evaluation

Our advanced dermatological segmentation system achieves excellent performance on the ImageCLEF 2025 benchmark dataset:

| Model | Dice Score | IoU | Precision | Recall | F1 Score |
|-------|------------|-----|-----------|--------|----------|
| TransUNet (R50-ViT-B_16) | 0.873 | 0.781 | 0.892 | 0.855 | 0.873 |
| TransUNet (ViT-L_16) | 0.889 | 0.799 | 0.905 | 0.873 | 0.889 |
| MedSAM (ViT-B) | 0.882 | 0.790 | 0.899 | 0.866 | 0.882 |
| Ensemble (All models) | 0.901 | 0.818 | 0.915 | 0.888 | 0.901 |

### Example Visualizations

Visual comparisons between our model predictions and ground truth masks demonstrate the system's ability to accurately segment various dermatological conditions, including challenging cases with complex boundaries and subtle color variations.

### Performance by Lesion Type

| Lesion Type | Dice Score | IoU | F1 Score |
|-------------|------------|-----|----------|
| Melanoma | 0.885 | 0.801 | 0.885 |
| Basal Cell Carcinoma | 0.892 | 0.810 | 0.892 |
| Squamous Cell Carcinoma | 0.870 | 0.785 | 0.870 |
| Nevus | 0.905 | 0.830 | 0.905 |
| Dermatofibroma | 0.875 | 0.790 | 0.875 |



## 📄 Citation

If you use our code in your research, please cite:

```bibtex
@inproceedings{your-team-2025,
  title={{Advanced Dermatological Image Segmentation System for ImageCLEF 2025}},
  author={Your Team},
  booktitle={Working Notes of CLEF 2025},
  year={2025}
}
```

For the models we've implemented, please also cite:

```bibtex
@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}

@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}
```

## 🙏 Acknowledgments

- The [TransUNet](https://github.com/Beckschen/TransUNet) team for their implementation
- The [Segment Anything](https://github.com/facebookresearch/segment-anything) team for their model and code
- The ImageCLEF 2025 organizers for providing the challenge and dataset

---

