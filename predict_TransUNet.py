#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import tifffile

# Import TransUNet from local directory
sys.path.append('./TransUNet')
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


def parse_args():
    parser = argparse.ArgumentParser(description='Predict with TransUNet for medical image segmentation')
    
    # Parameters for data
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images for prediction')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save binary mask predictions')
    
    # Parameters for model
    parser.add_argument('--checkpoint_paths', type=str, nargs='+', required=True,
                        help='List of checkpoint paths to use for prediction')
    parser.add_argument('--vit_names', type=str, nargs='+', required=True,
                        choices=['R50-ViT-B_16', 'ViT-B_16', 'ViT-B_32', 'ViT-L_32', 'ViT-L_16'],
                        help='ViT model names corresponding to checkpoints')
    parser.add_argument('--vit_patches_sizes', type=int, nargs='+', required=True,
                        help='Patch sizes corresponding to each model')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--n_classes', type=int, default=1,
                        help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for prediction')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu), defaults to available GPU if present')
    
    return parser.parse_args()


class PredictDataset(Dataset):
    """Dataset for loading images for prediction."""
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        # Load all images with jpg and png extensions
        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")) + 
                                  list(self.image_dir.glob("*.png")) +
                                  list(self.image_dir.glob("*.jpeg")))
        self.transform = transform
        print(f"Found {len(self.image_paths)} images for prediction.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Store original size (width, height)
        width, height = image.size
        
        # Convert to numpy array for transformation
        image_np = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            augmented = transform(image=image_np)
            image_tensor = augmented['image']
        
        # Return image tensor, path, and dimensions separately
        return image_tensor, str(img_path), width, height


def custom_collate_fn(batch):
    """Collate function to handle separate dimensions."""
    images = torch.stack([item[0] for item in batch])
    paths = [item[1] for item in batch]
    widths = [item[2] for item in batch]
    heights = [item[3] for item in batch]
    return images, paths, widths, heights


def get_predict_transform(img_size=224):
    """Get transforms for prediction."""
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return transform


def load_transunet_model(vit_name='R50-ViT-B_16',
                         img_size=224,
                         n_classes=1,
                         vit_patches_size=16,
                         pretrained_dir='pretrained_models',
                         load_pretrained=False):
    """Load TransUNet model without pretrained weights."""
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = n_classes
    if 'R50' in vit_name:
        config_vit.n_skip = 3
        config_vit.patches.grid = (
            img_size // vit_patches_size,
            img_size // vit_patches_size
        )
    else:
        config_vit.n_skip = 0
    model = ViT_seg(config_vit, img_size=img_size, num_classes=n_classes)
    if load_pretrained:
        pretrained_map = {
            'R50-ViT-B_16': 'imagenet21k_R50+ViT-B_16.npz',
            'ViT-B_16': 'imagenet21k_ViT-B_16.npz',
            'ViT-B_32': 'imagenet21k_ViT-B_32.npz',
            'ViT-L_32': 'imagenet21k_ViT-L_32.npz',
            'ViT-L_16': 'imagenet21k_ViT-L_16.npz'
        }
        if vit_name not in pretrained_map:
            raise ValueError(f"Selected {vit_name} but only supports: {list(pretrained_map.keys())}")
        pretrained_path = os.path.join(pretrained_dir, pretrained_map[vit_name])
        if os.path.exists(pretrained_path):
            weights = np.load(pretrained_path)
            model.load_from(weights)
            print(f"Loaded pretrained weights from {pretrained_path}")
        else:
            print(f"[Warning] Pretrained weights not found at {pretrained_path}.")
    else:
        print("Not loading pretrained weights. (Using checkpoint later)")
    return model


class TransUNetWrapper(nn.Module):
    """Wrapper model to ensure consistent output size."""
    def __init__(self, base_model, target_size=(224, 224)):
        super().__init__()
        self.base_model = base_model
        self.target_size = target_size

    def forward(self, x):
        out = self.base_model(x)
        if out.shape[2:] != self.target_size:
            out = F.interpolate(out, size=self.target_size, mode='bilinear', align_corners=True)
        return out


def predict_masks(model, loader, device, save_dir):
    """Predict binary masks and save as TIFF files."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    print("Starting mask prediction...")
    with torch.no_grad():
        for batch_idx, (images, img_paths, widths, heights) in enumerate(tqdm(loader, desc="Predicting")):
            images = images.to(device)
            outputs = model(images)
            
            # Ensure output size is correct
            if outputs.shape[2:] != (224, 224):
                outputs = F.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=True)
                
            # Generate binary predictions
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()  # shape: (B, 1, 224, 224)
            
            for i in range(preds.shape[0]):
                # Get dimensions for this image
                width = widths[i]
                height = heights[i]
                
                # Print debug info for first few images
                if batch_idx < 2:
                    print(f"Processing image {i} in batch {batch_idx}")
                    print(f"Image path: {img_paths[i]}")
                    print(f"Resizing to width={width}, height={height}")
                
                # Create mask and resize to original dimensions
                pred_mask = preds[i, 0].astype(np.uint8)
                mask_img = Image.fromarray(pred_mask)
                mask_resized = mask_img.resize((width, height), resample=Image.NEAREST)
                mask_resized = np.array(mask_resized)
                
                # Save the mask
                base_name = Path(img_paths[i]).stem
                save_path = os.path.join(save_dir, f"{base_name}_mask_sys.tiff")
                tifffile.imwrite(save_path, mask_resized)
                
                # Print confirmation for first few images
                if batch_idx < 2:
                    print(f"Saved mask to {save_path}")
    
    print(f"All mask predictions saved to {save_dir}")


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check that the number of model parameters match
    if not (len(args.checkpoint_paths) == len(args.vit_names) == len(args.vit_patches_sizes)):
        raise ValueError("The number of checkpoint paths, ViT names, and patches sizes must match.")
    
    # Create dataset and dataloader
    predict_tf = get_predict_transform(args.img_size)
    predict_dataset = PredictDataset(args.image_dir, transform=predict_tf)
    
    predict_loader = DataLoader(
        predict_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )    # Process each model configuration
    for idx, (checkpoint_path, vit_name, vit_patches_size) in enumerate(
        zip(args.checkpoint_paths, args.vit_names, args.vit_patches_sizes)
    ):
        # Create model-specific save directory
        model_save_dir = os.path.join(args.save_dir, f"predictions_{vit_name}")
        
        print("\n" + "="*50)
        print(f"Processing model {idx+1}/{len(args.checkpoint_paths)}: {vit_name}")
        print(f"Checkpoint path: {checkpoint_path}")
        print(f"Patch size: {vit_patches_size}")
        print("="*50)

        # Create base model
        base_model = load_transunet_model(
            vit_name=vit_name,
            img_size=args.img_size,
            n_classes=args.n_classes,
            vit_patches_size=vit_patches_size,
            load_pretrained=False
        ).to(device)

        # Wrap model
        model = TransUNetWrapper(base_model, target_size=(args.img_size, args.img_size)).to(device)

        # Load checkpoint
        if os.path.exists(checkpoint_path):
            try:
                # Try loading with weights_only=True (newer PyTorch versions)
                model.load_state_dict(
                    torch.load(checkpoint_path, map_location=device, weights_only=True)
                )
            except TypeError:
                # Fall back to old method
                model.load_state_dict(
                    torch.load(checkpoint_path, map_location=device)
                )
            print(f"Loaded trained checkpoint from {checkpoint_path}")
        else:
            print(f"[Warning] Checkpoint {checkpoint_path} not found. Skipping this model.")
            continue

        # Perform prediction - only output binary masks
        predict_masks(model, predict_loader, device, save_dir=model_save_dir)

    print("\nCompleted predictions for all models.")


if __name__ == "__main__":
    main()