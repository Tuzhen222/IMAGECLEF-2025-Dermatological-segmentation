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
import logging

# Import segmentation_models_pytorch for DiceLoss
import segmentation_models_pytorch as smp

# Import TransUNet from official repository
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

########################################
# 1. Custom Dataset Class
########################################
class CustomSegmentationDataset(Dataset):
    """
    Custom dataset class for medical image segmentation.
    
    Args:
        image_paths (list): List of directories containing images
        mask_paths (list): List of directories containing masks
        transform (callable, optional): Optional transform to be applied on a sample
    """
    
    def __init__(self, image_paths, mask_paths, transform=None):
        self.transform = transform
        
        # Collect images from all specified directories
        self.image_paths = []
        for img_dir in image_paths:
            img_dir = Path(img_dir)
            if img_dir.exists():
                self.image_paths.extend(img_dir.glob("*.jpg"))
                self.image_paths.extend(img_dir.glob("*.png"))
            else:
                logger.warning(f"Image directory {img_dir} does not exist")
        self.image_paths = sorted(self.image_paths)

        # Collect masks from all specified directories
        self.mask_paths = {}
        for mask_dir in mask_paths:
            mask_dir = Path(mask_dir)
            if mask_dir.exists():
                for mask_file in mask_dir.glob("*.tiff"):
                    stem = mask_file.stem
                    self.mask_paths[stem] = mask_file
            else:
                logger.warning(f"Mask directory {mask_dir} does not exist")

        # Keep only images that have corresponding masks
        self.image_paths = [p for p in self.image_paths if p.stem in self.mask_paths]
        logger.info(f"Found {len(self.image_paths)} images with corresponding masks")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[img_path.stem]

        # Load image
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            try:
                mask = np.array(Image.open(mask_path).convert("L"))
            except Exception as e:
                raise ValueError(f"Cannot read mask from {mask_path}") from e
        mask = (mask > 0).astype(np.uint8)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image_np, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask'].unsqueeze(0).float()
        else:
            # Default transforms if none provided
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

########################################
# 2. Transform Functions
########################################
def get_transforms(img_size=224, use_augmentation=True):
    """
    Create training and validation transforms.
    
    Args:
        img_size (int): Target image size
        use_augmentation (bool): Whether to use data augmentation for training
        
    Returns:
        tuple: (train_transform, validation_transform)
    """
    
    if use_augmentation:
        train_tf = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        train_tf = A.Compose([
            A.Resize(img_size, img_size),
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

########################################
# 3. Patch Size Utility
########################################
def get_patch_size(vit_name):
    """
    Get appropriate patch size for the given ViT model.
    
    Args:
        vit_name (str): Name of the ViT model
        
    Returns:
        int: Patch size
    """
    if '16' in vit_name:
        return 16
    elif '32' in vit_name:
        return 32
    else:
        return 16  # default

########################################
# 4. Model Loading Functions
########################################
def load_transunet_model(vit_name='R50-ViT-B_16',
                         img_size=224,
                         n_classes=1,
                         pretrained_dir='pretrained_models'):
    """
    Load TransUNet model with pretrained weights.
    
    Args:
        vit_name (str): Name of the ViT backbone
        img_size (int): Input image size
        n_classes (int): Number of output classes
        pretrained_dir (str): Directory containing pretrained weights
        
    Returns:
        torch.nn.Module: TransUNet model
    """
    
    # Automatically determine patch size based on model name
    vit_patches_size = get_patch_size(vit_name)
    logger.info(f"Using patch size: {vit_patches_size} for model {vit_name}")
    
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

    # Mapping of model names to pretrained files
    pretrained_map = {
        'R50-ViT-B_16': 'imagenet21k_R50+ViT-B_16.npz',
        'ViT-B_16': 'imagenet21k_ViT-B_16.npz',
        'ViT-B_32': 'imagenet21k_ViT-B_32.npz',
        'ViT-L_32': 'imagenet21k_ViT-L_32.npz',
        'ViT-L_16': 'imagenet21k_ViT-L_16.npz'
    }

    if vit_name not in pretrained_map:
        raise ValueError(f"Model {vit_name} not supported. Available models: {list(pretrained_map.keys())}")

    pretrained_path = os.path.join(pretrained_dir, pretrained_map[vit_name])
    if os.path.exists(pretrained_path):
        weights = np.load(pretrained_path)
        model.load_from(weights)
        logger.info(f"Loaded pretrained weights from {pretrained_path}")
    else:
        logger.warning(f"Pretrained file not found at {pretrained_path}. Training from scratch.")
        
    return model

########################################
# 5. Model Wrapper for Output Resizing
########################################
class TransUNetWrapper(nn.Module):
    """
    Wrapper class for TransUNet to handle output resizing.
    
    Args:
        base_model (torch.nn.Module): Base TransUNet model
        target_size (tuple): Target output size (height, width)
    """
    
    def __init__(self, base_model, target_size=(224, 224)):
        super().__init__()
        self.base_model = base_model
        self.target_size = target_size

    def forward(self, x):
        out = self.base_model(x)
        if out.shape[2:] != self.target_size:
            out = F.interpolate(out, size=self.target_size, mode='bilinear', align_corners=True)
        return out

########################################
# 6. Loss and Metrics
########################################
def dice_coeff(pred, target, smooth=1e-5):
    """
    Calculate Dice coefficient.
    
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        float: Dice coefficient
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

########################################
# 7. Training and Validation Functions
########################################
def train_epoch(model, loader, optimizer, criterion, device, scaler):
    """
    Train model for one epoch.
    
    Args:
        model (torch.nn.Module): Model to train
        loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        criterion: Loss function
        device (torch.device): Device to run on
        scaler (torch.cuda.amp.GradScaler): Mixed precision scaler
        
    Returns:
        tuple: (average_loss, average_dice_score)
    """
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

def validate_epoch(model, loader, criterion, device):
    """
    Validate model for one epoch.
    
    Args:
        model (torch.nn.Module): Model to validate
        loader (DataLoader): Validation data loader
        criterion: Loss function
        device (torch.device): Device to run on
        
    Returns:
        tuple: (average_loss, average_dice_score)
    """
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

########################################
# 8. Visualization Functions
########################################
def visualize_predictions(model, loader, device, num_images=3, save_path="predictions.png"):
    """
    Visualize model predictions.
    
    Args:
        model (torch.nn.Module): Trained model
        loader (DataLoader): Data loader
        device (torch.device): Device to run on
        num_images (int): Number of images to visualize
        save_path (str): Path to save the visualization
    """
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
                # Denormalize image
                img_np = img.cpu()[i].permute(1, 2, 0).numpy()
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np, 0, 1)
                
                images.append(img_np)
                masks.append(mask[i, 0].numpy())
                preds.append(pred.cpu()[i, 0].numpy())
                
                if len(images) >= num_images:
                    break
    
    # Create visualization
    fig, axs = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    if num_images == 1:
        axs = axs.reshape(1, -1)
        
    for i in range(num_images):
        axs[i, 0].imshow(images[i])
        axs[i, 0].set_title("Input Image")
        axs[i, 0].axis("off")
        
        axs[i, 1].imshow(masks[i], cmap="gray")
        axs[i, 1].set_title("Ground Truth")
        axs[i, 1].axis("off")
        
        axs[i, 2].imshow(preds[i], cmap="gray")
        axs[i, 2].set_title("Prediction")
        axs[i, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"Predictions saved to {save_path}")

def plot_training_history(history, save_path="training_history.png"):
    """
    Plot training history.
    
    Args:
        history (dict): Training history containing losses and metrics
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_dice'], label='Train Dice', marker='o')
    plt.plot(history['val_dice'], label='Val Dice', marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title("Dice Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"Training history saved to {save_path}")

########################################
# 9. Data Folder Configuration
########################################
def get_folder_groups():
    """
    Define different groups of augmentation folders.
    
    Returns:
        dict: Dictionary mapping group names to folder lists
    """
    all_subfolders = [
        "AdditiveNoise", "AdvancedAugmentation", "AdvancedAugmentation_2",
        "CenterCrop", "CompositeAug", "ElasticTransform", "GridDistortion",
        "HorizontalFlip", "Medium", "Medium_add_non_spatial_transformations",
        "OpticalDistortion", "Original", "PadIfNeeded", "RandomRotate90",
        "RandomSizedCrop", "RandomSizedCrop_0.1", "Transpose", "VerticalFlip",
        "gpt1", "gpt2", "gpt3"
    ]

    folder_groups = {
        "all": all_subfolders,
        "geometric": [
            "AdvancedAugmentation", "AdvancedAugmentation_2", "CenterCrop",
            "CompositeAug", "ElasticTransform", "GridDistortion", "HorizontalFlip",
            "Medium", "OpticalDistortion", "PadIfNeeded", "RandomRotate90",
            "RandomSizedCrop", "RandomSizedCrop_0.1", "Transpose", "VerticalFlip",
            "gpt1", "gpt2", "gpt3"
        ],
        "photometric": [
            "AdvancedAugmentation", "AdvancedAugmentation_2",
            "Medium_add_non_spatial_transformations", "gpt1", "gpt2", "gpt3"
        ],
        "noise_artifact": [
            "AdditiveNoise", "AdvancedAugmentation", "AdvancedAugmentation_2",
            "Medium_add_non_spatial_transformations", "gpt1", "gpt2"
        ],
        "original": ["Original"]
    }
    
    return folder_groups

########################################
# 10. Main Training Function
########################################
def main(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get folder groups and validate selection
    folder_groups = get_folder_groups()
    if args.folder_group not in folder_groups:
        logger.warning(f"Folder group '{args.folder_group}' not found. Using 'all' instead.")
        args.folder_group = "all"
    
    subfolders = folder_groups[args.folder_group]
    logger.info(f"Selected folder group '{args.folder_group}' with {len(subfolders)} folders")
    
    # Setup data paths
    image_train_paths = [os.path.join(args.train_images_path, folder) for folder in subfolders]
    label_train_paths = [os.path.join(args.train_labels_path, folder) for folder in subfolders]
    image_valid_paths = [args.val_images_path]
    label_valid_paths = [args.val_labels_path]
    
    # Create transforms
    train_tf, val_tf = get_transforms(args.img_size, args.use_augmentation)
    
    # Create datasets
    train_dataset = CustomSegmentationDataset(image_train_paths, label_train_paths, transform=train_tf)
    val_dataset = CustomSegmentationDataset(image_valid_paths, label_valid_paths, transform=val_tf)
    
    # Create data loaders
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
    
    # Load model
    logger.info(f"Loading TransUNet model: {args.model_name}")
    base_model = load_transunet_model(
        vit_name=args.model_name,
        img_size=args.img_size,
        n_classes=args.num_classes,
        pretrained_dir=args.pretrained_dir
    )
    model = TransUNetWrapper(base_model, target_size=(args.img_size, args.img_size)).to(device)
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            logger.info(f"Resumed from checkpoint {args.resume_from} at epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            logger.info(f"Loaded model weights from {args.resume_from}")
    
    # Setup training components
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=args.scheduler_patience, 
        factor=args.scheduler_factor, verbose=True
    )
    criterion = smp.losses.DiceLoss(mode='binary')
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_save_name = f"TransUNet_{args.model_name.replace('-', '_')}_{args.folder_group}"
    
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        
        # Validation
        val_loss, val_dice = validate_epoch(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, f"{model_save_name}_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice,
            }, best_model_path)
            logger.info(f"Saved best model to {best_model_path} with Val Loss = {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Save training history plot
    history_plot_path = os.path.join(args.output_dir, f"{model_save_name}_history.png")
    plot_training_history(history, history_plot_path)
    
    # Load best model and create predictions visualization
    best_model_path = os.path.join(args.output_dir, f"{model_save_name}_best.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        predictions_path = os.path.join(args.output_dir, f"{model_save_name}_predictions.png")
        visualize_predictions(model, val_loader, device, num_images=3, save_path=predictions_path)

########################################
# 11. Command Line Interface
########################################
def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="TransUNet Training Script for Medical Image Segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train_images_path', type=str, required=True,
                        help='Path to training images directory')
    parser.add_argument('--train_labels_path', type=str, required=True,
                        help='Path to training labels directory')
    parser.add_argument('--val_images_path', type=str, required=True,
                        help='Path to validation images directory')
    parser.add_argument('--val_labels_path', type=str, required=True,
                        help='Path to validation labels directory')
    parser.add_argument('--folder_group', type=str, default='all',
                        choices=['all', 'geometric', 'photometric', 'noise_artifact', 'original'],
                        help='Group of augmentation folders to use for training')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='R50-ViT-B_16',
                        choices=['R50-ViT-B_16', 'ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32'],
                        help='TransUNet model variant to use')
    parser.add_argument('--pretrained_dir', type=str, default='pretrained_models',
                        help='Directory containing pretrained model weights')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of output classes for segmentation')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (images will be resized to img_size x img_size)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--use_augmentation', action='store_true', default=False,
                        help='Use data augmentation during training')
    
    # Scheduler and early stopping
    parser.add_argument('--scheduler_patience', type=int, default=5,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor to reduce learning rate by')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Patience for early stopping')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save model checkpoints and plots')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    return parser.parse_args()

########################################
# 12. Entry Point
########################################
if __name__ == "__main__":
    args = parse_arguments()
    
    # Print configuration
    logger.info("Training Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    try:
        main(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise