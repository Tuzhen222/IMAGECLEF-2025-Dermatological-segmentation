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
import segmentation_models_pytorch as smp

# Import TransUNet from local directory
sys.path.append('./TransUNet')
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


def parse_args():
    parser = argparse.ArgumentParser(description='Train TransUNet for medical image segmentation')
    
    # Parameters for data
    parser.add_argument('--train_image_dirs', type=str, nargs='+', required=True,
                        help='List of directories containing training images')
    parser.add_argument('--train_mask_dirs', type=str, nargs='+', required=True,
                        help='List of directories containing training masks')
    parser.add_argument('--val_image_dirs', type=str, nargs='+', required=True,
                        help='List of directories containing validation images')
    parser.add_argument('--val_mask_dirs', type=str, nargs='+', required=True,
                        help='List of directories containing validation masks')
    
    # Parameters for model
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                        choices=['R50-ViT-B_16', 'ViT-B_16', 'ViT-B_32', 'ViT-L_32', 'ViT-L_16'],
                        help='ViT model name')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--n_classes', type=int, default=1,
                        help='Number of output classes')
    parser.add_argument('--vit_patches_size', type=int, default=16,
                        help='Size of ViT patches')
    parser.add_argument('--pretrained_dir', type=str, default='pretrained_models',
                        help='Directory containing pretrained weights')
    
    # Parameters for training
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--scheduler_patience', type=int, default=5,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor for learning rate scheduler')
    
    # Parameters for visualization
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for saving models and visualizations')
    parser.add_argument('--num_vis_images', type=int, default=3,
                        help='Number of images to visualize')
    
    # Miscellaneous parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision')
    
    return parser.parse_args()
class CustomSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.transform = transform
          # Collect images from all directories
        self.image_paths = []
        for img_dir in image_paths:
            img_dir = Path(img_dir)
            self.image_paths.extend(img_dir.glob("*.jpg"))
            self.image_paths.extend(img_dir.glob("*.png"))
        self.image_paths = sorted(self.image_paths)

        # Collect masks from all directories
        self.mask_paths = {}
        for mask_dir in mask_paths:
            mask_dir = Path(mask_dir)
            for mask_file in mask_dir.glob("*.tiff"):
                stem = mask_file.stem
                self.mask_paths[stem] = mask_file

        # Keep only images with corresponding masks
        self.image_paths = [p for p in self.image_paths if p.stem in self.mask_paths]
        print(f"Found {len(self.image_paths)} images with corresponding masks.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[img_path.stem]

        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            try:
                mask = np.array(Image.open(mask_path).convert("L"))
            except Exception as e:
                raise ValueError(f"Cannot read mask from {mask_path}") from e
        mask = (mask > 0).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image_np, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask'].unsqueeze(0).float()
        else:
            transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            augmented = transform(image=image_np, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask'].unsqueeze(0).float()
        return image_tensor, mask_tensor

def get_transforms(img_size=224):
    train_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_tf, val_tf

def load_transunet_model(vit_name='R50-ViT-B_16',
                         img_size=224,
                         n_classes=1,
                         vit_patches_size=16,
                         pretrained_dir='pretrained_models'):
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

    pretrained_map = {
        'R50-ViT-B_16': 'imagenet21k_R50+ViT-B_16.npz',
        'ViT-B_16': 'imagenet21k_ViT-B_16.npz',
        'ViT-B_32': 'imagenet21k_ViT-B_32.npz',
        'ViT-L_32': 'imagenet21k_ViT-L_32.npz',
        'ViT-L_16': 'imagenet21k_ViT-L_16.npz'
    }

    if vit_name not in pretrained_map:
        raise ValueError(f"You selected {vit_name} but only these are supported: {list(pretrained_map.keys())}")

    pretrained_path = os.path.join(pretrained_dir, pretrained_map[vit_name])
    if os.path.exists(pretrained_path):
        weights = np.load(pretrained_path)
        model.load_from(weights)        
        print(f"Loaded pretrained weights from {pretrained_path}")
    else:
        print(f"[Warning] Pretrained file not found at {pretrained_path}. Model will train from scratch.")
    return model

class TransUNetWrapper(nn.Module):
    def __init__(self, base_model, target_size=(224,224)):
        super().__init__()
        self.base_model = base_model
        self.target_size = target_size

    def forward(self, x):
        out = self.base_model(x)
        if out.shape[2:] != self.target_size:
            out = F.interpolate(out, size=self.target_size, mode='bilinear', align_corners=True)
        return out

def dice_coeff(pred, target):
    smooth = 1e-5
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train(model, loader, optimizer, criterion, device, scaler):
    model.train()
    epoch_loss = 0
    dice_scores = []
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        dice_scores.append(dice_coeff(outputs, masks).item())
    return epoch_loss / len(loader), sum(dice_scores) / len(dice_scores)

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    dice_scores = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            val_loss += loss.item()
            dice_scores.append(dice_coeff(outputs, masks).item())
    return val_loss / len(loader), sum(dice_scores) / len(dice_scores)

def visualize_predictions(model, loader, device, num_images=3, output_dir='.'):
    model.eval()
    images, masks, preds = [], [], []
    with torch.no_grad():
        for img, mask in loader:
            if len(images) >= num_images:
                break
            img = img.to(device)
            with torch.cuda.amp.autocast():
                pred = torch.sigmoid(model(img)) > 0.5
            for i in range(min(img.size(0), num_images - len(images))):
                img_np = img.cpu()[i].permute(1,2,0).numpy()
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np, 0, 1)
                images.append(img_np)
                masks.append(mask[i,0].numpy())
                preds.append(pred.cpu()[i,0].numpy())
                if len(images) >= num_images:
                    break
    fig, axs = plt.subplots(num_images, 3, figsize=(12, 4*num_images))
    for i in range(num_images):
        axs[i,0].imshow(images[i])
        axs[i,0].set_title("Input Image")
        axs[i,0].axis("off")
        axs[i,1].imshow(masks[i], cmap="gray")
        axs[i,1].set_title("Ground Truth")
        axs[i,1].axis("off")
        axs[i,2].imshow(preds[i], cmap="gray")
        axs[i,2].set_title("Prediction")
        axs[i,2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "TransUNet_prediction_results.png"))
    plt.close()

def plot_training_history(history, output_dir='.'):
    plt.figure(figsize=(18,6))
    plt.subplot(1,3,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.plot(history['val_dice'], label='Val Dice')
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Dice Coefficient")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "TransUNet_training_history.png"))
    plt.close()



def main():
    args = parse_args()
      # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
      # Get transforms
    train_tf, val_tf = get_transforms(img_size=args.img_size)
      # Create datasets
    train_dataset = CustomSegmentationDataset(args.train_image_dirs, args.train_mask_dirs, transform=train_tf)
    val_dataset = CustomSegmentationDataset(args.val_image_dirs, args.val_mask_dirs, transform=val_tf)
      # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
      # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      # Create model
    base_model = load_transunet_model(
        vit_name=args.vit_name,
        img_size=args.img_size,
        n_classes=args.n_classes,
        vit_patches_size=args.vit_patches_size,
        pretrained_dir=args.pretrained_dir
    )
    model = TransUNetWrapper(base_model, target_size=(args.img_size, args.img_size)).to(device)
      # Set up optimizer, scheduler and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 
        patience=args.scheduler_patience, 
        factor=args.scheduler_factor, 
        verbose=True
    )
    criterion = smp.losses.DiceLoss(mode='binary')
      # Set up AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
      # Tracking for early stopping and training history
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Training
        train_loss, train_dice = train(model, train_loader, optimizer, criterion, device, scaler)
        
        # Validation
        val_loss, val_dice = validate(model, val_loader, criterion, device)
          # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
          # Print results
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss:   {val_loss:.4f}, Val Dice:   {val_dice:.4f}")
          # Update learning rate
        scheduler.step(val_loss)
          # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.output_dir, "TransUNet_best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f">> Saved best model with Val Loss = {best_val_loss:.4f} to {model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
          # Early stopping
        if patience_counter >= args.patience:
            print(f">> Early stopping after {epoch+1} epochs")
            break
      # Plot training history
    plot_training_history(history, args.output_dir)
    
    # Load best model and visualize results
    best_model_path = os.path.join(args.output_dir, "TransUNet_best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    visualize_predictions(model, val_loader, device, num_images=args.num_vis_images, output_dir=args.output_dir)
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()