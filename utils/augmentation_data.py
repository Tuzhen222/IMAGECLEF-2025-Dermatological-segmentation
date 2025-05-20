#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Augmentation Script for IMAGECLEF-2025-Dermatological-segmentation
This script applies various augmentation techniques to images and their corresponding masks
for dermatological segmentation tasks.
"""

import random
import cv2
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import glob
import albumentations as A
import argparse
import sys

# Set random seed for reproducibility
random.seed(42)

def get_augmentation(method, original_height, original_width):
    """Return augmentation transform based on method name"""

    if method == "RandomRotate90":
        return A.RandomRotate90(p=1)
    
    elif method == "HorizontalFlip":
        return A.HorizontalFlip(p=1)
    
    elif method == "GridDistortion":
        return A.GridDistortion(p=1)
    
    elif method == "VerticalFlip":
        return A.VerticalFlip(p=1)
    
    elif method == "ElasticTransform":
        return A.ElasticTransform(p=1)
    
    elif method == "Medium_add_non_spatial__stranformations":
        return A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            ], p=0.7),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
                A.ImageCompression(quality_lower=75, quality_upper=100, p=0.5),
            ], p=0.5),
        ])
    
    elif method == "Medium":
        return A.Compose([
            A.OneOf([
                A.RandomSizedCrop(min_max_height=(int(0.5*original_height), original_height), 
                                size=(original_height, original_width), p=0.5),
                A.PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)
            ],p=1),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
            ], p=0.8)])
    
    elif method == "gpt1":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.2, p=0.5),
                A.ElasticTransform(alpha=50, sigma=50 * 0.5, alpha_affine=50 * 0.5, p=0.5)
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2.0, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=5, p=0.5),
                A.GaussNoise(var_limit=(20.0, 70.0), p=0.5),
            ], p=0.5),
            A.CoarseDropout(max_holes=8, max_height=int(0.05*original_height), 
                          max_width=int(0.05*original_width),
                          fill_value=0, mask_fill_value=0, p=0.4)
        ])
    
    elif method == "gpt2":
        return A.Compose([
            A.RandomCrop(height=int(0.8*original_height), width=int(0.8*original_width), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.2, p=0.5),
                A.ElasticTransform(alpha=50, sigma=50 * 0.5, alpha_affine=50 * 0.5, p=0.5)
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2.0, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=5, p=0.5),
                A.GaussNoise(var_limit=(20.0, 70.0), p=0.5),
            ], p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                          fill_value=0, mask_fill_value=0, p=0.4)
        ])
    
    elif method == "gpt3":
        return A.Compose([
            A.RandomResizedCrop(size=(original_height, original_width), scale=(0.8, 1.0), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.CLAHE(p=0.5),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5)
        ])
    elif method == "PadIfNeeded":
        return A.PadIfNeeded(min_height=original_height, min_width=original_width, border_mode=cv2.BORDER_CONSTANT, value=0)
    elif method == "AdditiveNoise":
        return A.GaussNoise(var_limit=(10.0, 50.0), p=1)
    elif method == "AdvancedAugmentation":
        return A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.6),
                A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.6),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, 
                    scale_limit=0.1, 
                    rotate_limit=20, 
                    border_mode=cv2.BORDER_CONSTANT, 
                    value=0,
                    p=0.6
                ),
            ], p=0.7),
            
            # Elastic deformations - simulates skin elasticity and natural variations
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.6),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.6),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5)
            ], p=0.5),
            
            # Lighting and color variations - simulates different clinical imaging conditions
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5)
            ], p=0.6),
            
            # Blur and sharpness variations - simulates focus issues in clinical photography
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=0.4),
                A.MotionBlur(blur_limit=3, p=0.4),
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=0.5)
            ], p=0.4),
            
            # Simulation of noise and artifacts - common in clinical photography
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.5),
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.5)
            ], p=0.4),
            
            # Cutouts - helps the model focus on different sections of lesions
            A.OneOf([
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.5),
                A.GridDropout(ratio=0.1, unit_size_min=8, unit_size_max=16, random_offset=True, p=0.4)
            ], p=0.3),
            
            # Subtle color adjustments - helps with different skin tones and lighting conditions
            A.OneOf([
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
                A.ChannelShuffle(p=0.3),
                A.ToSepia(p=0.3)
            ], p=0.4),
        ], p=1.0)

    elif method == "AdvancedAugmentation_2":
        return A.Compose([
            # OneOf spatial transformations with fixed parameters
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=1.0,
                                 border_mode=cv2.BORDER_CONSTANT, value=0),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
                A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=1.0)
            ], p=0.8),
            
            # Flips - anatomically plausible for many medical imaging contexts
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            
            # Pixel-level augmentations with fixed parameters
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.7),
            
            # Simulation of medical imaging artifacts with fixed parameters
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.4),
            
            # Cutout with fixed parameters
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, mask_fill_value=0, p=0.3),
        ])
    
    elif method == "CenterCrop":
        min_height = int(original_height * 0.1)
        return A.CenterCrop(height=min_height, width=original_width, p=0.9) 
    
    elif method == "CompositeAug":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.5,
                              border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.5),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.5)
        ], p=1)
    
    elif method == "OpticalDistortion":
        return A.OpticalDistortion(
         distort_limit=0.2, shift_limit=0.2, p=0.9)
    
    elif method == "RandomSizeCrop":
        min_height = int(original_height * 0.4)
        return A.RandomSizedCrop(
            min_max_height=(int(0.4 * original_height), original_height),
            height=min_height,
            width=original_width,
            p=0.9
        )
    elif method == "RandomSizedCrop_0.1":
        min_height = int(original_height * 0.1)
        return A.RandomSizedCrop(
            min_max_height=(int(0.1 * original_height), original_height),
            height=min_height,
            width=original_width,
            p=0.9
        )
    elif method == "Transpose":
        return A.Transpose(p=0.9)
    else:
        raise ValueError(f"Unknown augmentation method: {method}")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Image augmentation with various methods")
parser.add_argument("--method", type=str, required=True,
                   choices=["AdditiveNoise", "AdvancedAugmentation", "AdvancedAugmentation_2", "CenterCrop", 
                            "CompositeAug", "ElasticTransform", "GridDistortion", "HorizontalFlip", 
                            "Medium", "Medium_add_non_spatial__stranformations", "OpticalDistortion", 
                            "PadIfNeeded", "RandomRotate90", "RandomSizeCrop", "RandomSizedCrop_0.1", 
                            "Transpose", "VerticalFlip", "gpt1", "gpt2", "gpt3"],
                   help="Augmentation method to use")
parser.add_argument("--base_dir", type=str, default="../data",
                   help="Base directory containing images and labels")

args = parser.parse_args()

# Update global variables with arguments
method = args.method
base_dir = args.base_dir

# Define paths
train_dir = os.path.join(base_dir, "images_train")

# Define mask directories
mask_train_dir = os.path.join(base_dir, "labels_train")

# Create output directories
output_base = f"aug_dataset/{method}"
os.makedirs(output_base, exist_ok=True)

for folder in ["images_train"]:
    os.makedirs(os.path.join(output_base, folder), exist_ok=True)

# Create output directories for masks
for folder in ["labels_train"]:
    os.makedirs(os.path.join(output_base, folder), exist_ok=True)

# Function to find all mask files for a given image
def find_masks(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Determine if this is from train or valid set
    if "images_train" in image_path:
        mask_dir = mask_train_dir
    else:
        print(f"Invalid image path, not in train directory: {image_path}")
        return []

    mask_pattern = f"{base_name}.tiff"
    mask_paths = glob.glob(os.path.join(mask_dir, mask_pattern))
    return mask_paths

# Function to process a single image and its masks
def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find and read all masks for this image
    mask_paths = find_masks(image_path)
    if not mask_paths:
        print(f"No mask files found for: {image_path}")
        return
    
    # Get original dimensions for augmentations
    original_height, original_width = image.shape[:2]
    
    # Get augmentation transform based on method
    aug = get_augmentation(method, original_height, original_width)

    masks = []
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Could not read mask: {mask_path}")
            continue
        masks.append(mask)

    if not masks:
        print(f"No valid masks for: {image_path}")
        return

    # Apply augmentation to image and all masks
    aug_data = {"image": image}
    for i, mask in enumerate(masks):
        aug_data[f"mask{i}"] = mask

    # Create additional targets for each mask
    additional_targets = {f"mask{i}": "mask" for i in range(len(masks))}
    augmentation = A.Compose([aug], additional_targets=additional_targets, is_check_shapes=False)

    # Apply augmentation
    augmented = augmentation(**aug_data)

    # Get the augmented image and masks
    augmented_image = augmented["image"]
    augmented_masks = [augmented[f"mask{i}"] for i in range(len(masks))]

    # Save augmented image
    output_folder = os.path.dirname(image_path).replace(base_dir, output_base)
    os.makedirs(output_folder, exist_ok=True)

    img_base, img_ext = os.path.splitext(os.path.basename(image_path))
    output_image_path = os.path.join(output_folder, f"{img_base}_{method}{img_ext}")
    cv2.imwrite(output_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

    # Save augmented masks
    for mask_path, augmented_mask in zip(mask_paths, augmented_masks):
        # Determine if this is from train or valid
        if "train" in mask_path:
            mask_folder = os.path.join(output_base, "labels_train")
        else:
            mask_folder = os.path.join(output_base, "labels_valid")

        os.makedirs(mask_folder, exist_ok=True)
        mask_base, mask_ext = os.path.splitext(os.path.basename(mask_path))
        output_mask_path = os.path.join(mask_folder, f"{mask_base}_{method}{mask_ext}")
        cv2.imwrite(output_mask_path, augmented_mask)

# Process images from train set only
folders = [train_dir]

for folder in folders:
    if not os.path.exists(folder):
        print(f"Skipping {folder} as it doesn't exist")
        continue

    print(f"Processing images in {folder}")
    jpg_images = glob.glob(os.path.join(folder, "*.jpg"))
    png_images = glob.glob(os.path.join(folder, "*.png"))
    image_paths = jpg_images + png_images
    
    if not image_paths:
        print(f"No images found in {folder}. Please check the --base_dir argument.")
        sys.exit(1)

    for image_path in tqdm(image_paths):
        process_image(image_path)

print("\nAugmentation completed successfully!")
print(f"Augmented images and masks are saved in: {output_base}")
