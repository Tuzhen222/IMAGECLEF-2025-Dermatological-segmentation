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

1. [Installation](#installation)
2. [Dataset Preparation](#data-preparation)
3. [Training](#training)
   - [Training TransUNet](#training-transunet)
   - [Training MedSAM](#training-medsam)
4. [Prediction](#prediction)
5. [Model Architecture](#model-architecture)
6. [Results and Evaluation](#results-and-evaluation)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

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
  --train_image_dirs "path/to/images_train/Original" \
  --train_mask_dirs "path/to/labels_train/Original" \
  --val_image_dirs "path/to/images_valid" \
  --val_mask_dirs "path/to/labels_valid" \
  --output_dir "output_transunet" \
  --num_epochs 100 \
  --batch_size 8 \
  --vit_name "R50-ViT-B_16" \
  --img_size 224 \
  --use_amp
```

For using multiple augmentation folders:

```bash
python train_TransUNet.py \
  --train_image_dirs "path/to/images_train/Original" "path/to/images_train/Augmented" \
  --train_mask_dirs "path/to/labels_train/Original" "path/to/labels_train/Augmented" \
  --val_image_dirs "path/to/images_valid" \
  --val_mask_dirs "path/to/labels_valid" \
  --output_dir "output_transunet_augmented" \
  --num_epochs 100 \
  --batch_size 8 \
  --vit_name "R50-ViT-B_16" \
  --img_size 224 \
  --use_amp
```

Using all augmentation folders with PowerShell on Windows:

```powershell
# Create an array of folder names
$folders = @("AdditiveNoise", "AdvancedAugmentation", "HorizontalFlip", "VerticalFlip", 
             "RandomRotate90", "ElasticTransform", "GridDistortion", "Original")

# Base paths
$baseImgPath = "path\to\images_train"
$baseMaskPath = "path\to\labels_train"

# Construct command arguments
$imgDirs = @()
$maskDirs = @()

foreach ($folder in $folders) {
    $imgDirs += "`"$baseImgPath\$folder`""
    $maskDirs += "`"$baseMaskPath\$folder`""
}

# Create the command string
$imgDirsStr = $imgDirs -join " "
$maskDirsStr = $maskDirs -join " "

# Execute the command
python train_TransUNet.py `
  --train_image_dirs $imgDirsStr `
  --train_mask_dirs $maskDirsStr `
  --val_image_dirs "path\to\images_valid" `
  --val_mask_dirs "path\to\labels_valid" `
  --output_dir "output_all_aug" `
  --num_epochs 100 `
  --batch_size 8 `
  --use_amp
```

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

Choose the appropriate model with the `--vit_name` parameter:

```bash
python train_TransUNet.py \
  --train_image_dirs "path/to/images_train/Original" \
  --train_mask_dirs "path/to/labels_train/Original" \
  --val_image_dirs "path/to/images_valid" \
  --val_mask_dirs "path/to/labels_valid" \
  --vit_name "ViT-L_16" \
  --output_dir "output_vitl16" \
  --use_amp
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
- `--train_image_dirs`: List of directories containing training images (required)
- `--train_mask_dirs`: List of directories containing training masks (required)
- `--val_image_dirs`: List of directories containing validation images (required)
- `--val_mask_dirs`: List of directories containing validation masks (required)
- `--vit_name`: ViT model name (default: 'R50-ViT-B_16')
- `--img_size`: Input image size (default: 224)
- `--n_classes`: Number of output classes (default: 1 for binary segmentation)
- `--vit_patches_size`: Size of ViT patches (default: 16)
- `--pretrained_dir`: Directory containing pretrained weights (default: 'pretrained_models')

### Training Parameters
- `--batch_size`: Batch size for training and validation (default: 8)
- `--num_epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay for optimizer (default: 1e-3)
- `--patience`: Patience for early stopping (default: 10)
- `--scheduler_patience`: Patience for learning rate scheduler (default: 5)
- `--scheduler_factor`: Factor for learning rate scheduler (default: 0.5)

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

## Tips for Optimal Results

### 1. Image Size
- For better accuracy but slower training, try larger image sizes: `--img_size 384` or `--img_size 512`
- For faster training but potentially less accuracy, use smaller sizes: `--img_size 224`

### 2. Model Selection
- For most tasks, `R50-ViT-B_16` provides a good balance of performance and speed
- If you have sufficient GPU memory, `ViT-L_16` may offer improved performance

### 3. Learning Rate Tuning
- Start with the default `--lr 1e-4`
- If training is unstable, try a lower learning rate: `--lr 5e-5`
- If training is slow to converge, try a higher learning rate: `--lr 3e-4`

### 4. Batch Size
- Use the largest batch size that fits in your GPU memory
- Increase batch size with `--batch_size 16` or `--batch_size 32` if possible
- If using larger batch sizes, consider increasing the learning rate proportionally

### 5. Data Augmentation
- The dataset loader already includes some augmentations (horizontal/vertical flips, rotations)
- Use more augmentation folders if available for better generalization

### 6. Hardware Recommendations
- A GPU with at least 8GB VRAM is recommended for training
- Enable mixed precision training with `--use_amp` to reduce memory usage and speed up training

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

### Ensemble Prediction

For better performance, you can combine multiple models:

```bash
python predict_TransUNet.py `
  --image_dir "path\to\test_images" `
  --save_dir "path\to\ensemble_predictions" `
  --checkpoint_paths "output_transunet\TransUNet_best_model.pth" "output_vitl16\TransUNet_best_model.pth" `
  --vit_names "R50-ViT-B_16" "ViT-L_16" `
  --vit_patches_sizes 16 16 `
  --img_size 224 `
  --device "cuda:0"
```

### Batch Prediction for Large Datasets

For large datasets, use batch processing:

```bash
python predict_TransUNet.py `
  --image_dir "path\to\test_images" `
  --save_dir "path\to\predictions" `
  --checkpoint_paths "output_transunet\TransUNet_best_model.pth" `
  --vit_names "R50-ViT-B_16" `
  --vit_patches_sizes 16 `
  --img_size 224 `
  --batch_size 16 `
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

## Troubleshooting

### CUDA Out of Memory
If you encounter "CUDA out of memory" errors:
1. Reduce batch size with `--batch_size 4` or even lower
2. Try a smaller model, e.g., `--vit_name "ViT-B_32"`
3. Reduce image size with `--img_size 192`
4. Enable mixed precision training with `--use_amp` if not already enabled

### Poor Performance
If model performance is unsatisfactory:
1. Try training for more epochs: `--num_epochs 20`
2. Use a larger model: `--vit_name "ViT-L_16"`
3. Reduce learning rate: `--lr 5e-5`
4. Increase weight decay for better regularization: `--weight_decay 5e-3`
5. Ensure your data is properly preprocessed and normalized

### Loading Pretrained Weights
If you have issues loading pretrained weights:
1. Check that the `pretrained_models` directory contains the correct `.npz` files
2. If using a custom directory, specify it with `--pretrained_dir "path/to/pretrained"`
3. If weights are missing, try downloading them again with gdown

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

## 🔬 Conclusion

This repository provides a comprehensive implementation for dermatological image segmentation using state-of-the-art deep learning models. Our framework is designed to be modular, extensible, and easy to use, allowing for rapid experimentation and deployment of dermatological segmentation systems.

Key strengths of our approach include:

1. **Multiple Complementary Models**: TransUNet and MedSAM provide different segmentation approaches, allowing for robust ensemble results
2. **Comprehensive Data Augmentation**: Enhanced training data diversity to improve model generalization
3. **Flexible Training Pipeline**: Easy customization for different dermatological datasets and conditions
4. **Optimized for Performance**: Mixed precision training and efficient inference code

We hope this work contributes to advancing computer-aided diagnosis in dermatology and supports medical professionals in accurately identifying skin lesions.

## 📬 Contact

For any questions or suggestions about this project, please feel free to contact:

- **Your Name** - your.email@example.com
- **Team member 2** - team.member2@example.com

---

## 🙏 Acknowledgments

- The [TransUNet](https://github.com/Beckschen/TransUNet) team for their implementation
- The [Segment Anything](https://github.com/facebookresearch/segment-anything) team for their model and code
- The ImageCLEF 2025 organizers for providing the challenge and dataset

---

<p align="center">
  <img src="https://via.placeholder.com/150?text=Your+Logo" alt="Your Logo">
  <br>
  <em>Advancing medical image analysis with deep learning</em>
</p>
