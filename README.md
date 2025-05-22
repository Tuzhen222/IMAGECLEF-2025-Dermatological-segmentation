# IMAGECLEF-MEDIQA-MAGIC-2025-Dermatological-segmentation

This repository contains code for training and evaluating TransUNet models for medical image segmentation. TransUNet combines CNN features with the self-attention mechanism of Vision Transformers (ViT) to achieve excellent performance on segmentation tasks.

## Table of Contents
1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Model Configurations](#model-configurations)
5. [Training Parameters](#training-parameters)
6. [Output](#output)
7. [Tips for Optimal Results](#tips-for-optimal-results)
8. [Troubleshooting](#troubleshooting)

## Installation

### Step 1: Clone this repository
```bash
git clone https://github.com/Zhennor/IMAGECLEF-2025-Dermatological-segmentation
cd IMAGECLEF-2025-Dermatological-segmentation
```

### Step 2: Create a virtual environment (recommended)
```bash
# Create a virtual environment
python -m venv transunet_env

# Activate the environment
# On Windows:
transunet_env\Scripts\activate
# On macOS/Linux:
source transunet_env/bin/activate
```

### Step 3: Install required packages
```bash
pip install -r requirements.txt
```

### Step 4: Clone the TransUNet repository
```bash
git clone https://github.com/Beckschen/TransUNet.git
pip install -r TransUNet/requirements.txt
```

### Step 5: Download pretrained models

You can download the official pretrained weights for ViT models directly from Google Cloud:

🔗 [Official Google Cloud Storage](https://console.cloud.google.com/storage/browser/vit_models?inv=1&invt=AbyCpw)

Alternatively, for convenience, the weights have also been mirrored to Google Drive:

🔗 [Google Drive Mirror](https://drive.google.com/file/d/1mGhBIDcyollAEz0JMafXUO5zO3hErq4H/view?usp=sharing)
```bash
# Install gdown if not installed yet
pip install gdown

# Download pretrained weights
gdown "https://drive.google.com/file/d/1mGhBIDcyollAEz0JMafXUO5zO3hErq4H/view?usp=sharing" --fuzzy

# Extract the downloaded weights
mkdir -p pretrained_models
unzip pretrained_models.zip -d pretrained_models
```

## Data Preparation

Organize your dataset with the following structure:

```
dataset/
  ├── images_train/
  │    ├── AdditiveNoise/             # Augmentation folder 1
  │    │    ├── image1.jpgs
  │    │    ├── image2.jpgs
  │    │    └── ...
  │    ├── AdvancedAugmentation/      # Augmentation folder 2
  │    │    └── ...
  │    ├── ...                        # Other augmentation folders
  │    └── Original/                  # Original images
  │         └── ...
  ├── labels_train/
  │    ├── AdditiveNoise/             # Masks for augmentation folder 1
  │    │    ├── image1.tiff
  │    │    ├── image2.tiff
  │    │    └── ...
  │    ├── AdvancedAugmentation/      # Masks for augmentation folder 2 
  │    │    └── ...
  │    ├── ...                        # Masks for other augmentation folders
  │    └── Original/                  # Masks for original images
  │         └── ...
  ├── images_valid/                   # Validation images
  │    ├── image1.jpgs
  │    ├── image2.jpgs
  │    └── ...
  └── labels_valid/                   # Validation masks
       ├── image1.tiff
       ├── image2.tiff
       └── ...
```

**Important Notes:**
- The dataset loader matches images and masks by their filename stems (without extension)
- Supported image formats: .jpg, .png, supported mask formats: .tiff

## Training

### Basic Training Command

```bash
python transunet_segmentation.py \
  --train_image_dirs "path/to/images_train/Original" \
  --train_mask_dirs "path/to/labels_train/Original" \
  --val_image_dirs "path/to/images_valid" \
  --val_mask_dirs "path/to/labels_valid" \
  --output_dir "output" \
  --num_epochs 10 \
  --batch_size 8 \
  --use_amp
```

### Using Multiple Augmentation Folders

To train with multiple augmentation folders, specify each folder separately:

```bash
python transunet_segmentation.py \
  --train_image_dirs "path/to/images_train/Original" "path/to/images_train/AdditiveNoise" "path/to/images_train/HorizontalFlip" \
  --train_mask_dirs "path/to/labels_train/Original" "path/to/labels_train/AdditiveNoise" "path/to/labels_train/HorizontalFlip" \
  --val_image_dirs "path/to/images_valid" \
  --val_mask_dirs "path/to/labels_valid" \
  --output_dir "output_multi_aug" \
  --num_epochs 10 \
  --batch_size 8 \
  --use_amp
```

### Using All Augmentation Folders

For convenience, you can use a shell script to include all augmentation folders:

```bash
#!/bin/bash

# List of all augmentation folders
FOLDERS=("AdditiveNoise" "AdvancedAugmentation" "AdvancedAugmentation_2" "CenterCrop" 
         "CompositeAug" "ElasticTransform" "GridDistortion" "HorizontalFlip" 
         "Medium" "Medium_add_non_spatial_transformations" "OpticalDistortion" 
         "Original" "PadIfNeeded" "RandomRotate90" "RandomSizedCrop" 
         "RandomSizedCrop_0.1" "Transpose" "VerticalFlip" "gpt1" "gpt2" "gpt3")

# Base paths
BASE_IMG_PATH="path/to/images_train"
BASE_MASK_PATH="path/to/labels_train"

# Construct command arguments
IMG_DIRS=""
MASK_DIRS=""

for FOLDER in "${FOLDERS[@]}"; do
    IMG_DIRS="${IMG_DIRS} \"${BASE_IMG_PATH}/${FOLDER}\""
    MASK_DIRS="${MASK_DIRS} \"${BASE_MASK_PATH}/${FOLDER}\""
done

# Execute the command
eval "python transunet_segmentation.py \
  --train_image_dirs ${IMG_DIRS} \
  --train_mask_dirs ${MASK_DIRS} \
  --val_image_dirs \"path/to/images_valid\" \
  --val_mask_dirs \"path/to/labels_valid\" \
  --output_dir \"output_all_aug\" \
  --num_epochs 10 \
  --batch_size 8 \
  --use_amp"
```

Save this script as `train_all.sh`, make it executable with `chmod +x train_all.sh`, and run it with `./train_all.sh`.

## Model Configurations

The script supports several TransUNet model configurations:

| Model Name | Description | Size | Speed | Memory |
|------------|-------------|------|-------|--------|
| R50-ViT-B_16 | ResNet50 + ViT-Base with 16×16 patches | Medium | Fast | Medium |
| ViT-B_16 | ViT-Base with 16×16 patches | Medium | Medium | Medium |
| ViT-B_32 | ViT-Base with 32×32 patches | Medium | Fast | Low |
| ViT-L_16 | ViT-Large with 16×16 patches | Large | Slow | High |
| ViT-L_32 | ViT-Large with 32×32 patches | Large | Medium | Medium |

Specify the model with the `--vit_name` parameter:

```bash
python transunet_segmentation.py \
  --train_image_dirs "path/to/images_train/Original" \
  --train_mask_dirs "path/to/labels_train/Original" \
  --val_image_dirs "path/to/images_valid" \
  --val_mask_dirs "path/to/labels_valid" \
  --vit_name "ViT-L_16" \
  --output_dir "output_vitl16" \
  --use_amp
```

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

---

For more information, refer to the [TransUNet paper](https://arxiv.org/abs/2102.04306) and the [official repository](https://github.com/Beckschen/TransUNet).
