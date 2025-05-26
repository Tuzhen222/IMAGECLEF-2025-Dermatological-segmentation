import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
import random
import glob
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from skimage import transform
from segment_anything import sam_model_registry

# Set random seeds for reproducibility
torch.manual_seed(2023)
torch.cuda.empty_cache()

# Set environment variables for optimal performance
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"


def parse_arguments():
    """Parse command line arguments for training MedSAM."""
    parser = argparse.ArgumentParser(
        description="Train MedSAM for medical image segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to training npy files directory (contains 'gts' and 'imgs' subfolders)"
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        required=True,
        help="Path to validation npy files directory (contains 'gts' and 'imgs' subfolders)"
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="./work_dir",
        help="Working directory to save models and logs"
    )
    
    # Model configuration
    parser.add_argument(
        "--task_name",
        type=str,
        default="MedSAM-ViT-B",
        help="Task name for experiment tracking"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model type"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SAM pretrained checkpoint"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint for resuming training"
    )
    
    # Training parameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-7,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer"
    )
    
    # Loss function
    parser.add_argument(
        "--loss_type",
        type=str,
        default="iou",
        choices=["iou", "focal", "bce"],
        help="Loss function type"
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Alpha parameter for focal loss"
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss"
    )
    
    # Dataset parameters
    parser.add_argument(
        "--bbox_shift",
        type=int,
        default=20,
        help="Random shift for bounding box augmentation"
    )
    parser.add_argument(
        "--min_appearance",
        type=int,
        default=2,
        help="Minimum appearances across annotators for consensus"
    )
    
    # Training options
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision training"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for training"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="Save model every N epochs"
    )
    
    # Visualization and logging
    parser.add_argument(
        "--visualize_data",
        action="store_true",
        help="Create data visualization plots"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        default="",
        help="Weights & Biases API key"
    )
    
    return parser.parse_args()


def show_mask(mask, ax, random_color=False):
    """
    Visualize segmentation mask on matplotlib axis.
    
    Args:
        mask (numpy.ndarray): Binary mask to visualize
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on
        random_color (bool): Use random color for mask
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    """
    Visualize bounding box on matplotlib axis.
    
    Args:
        box (numpy.ndarray): Bounding box coordinates [x_min, y_min, x_max, y_max]
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


class NpyDataset(Dataset):
    """
    Dataset class for loading preprocessed medical images and multi-annotator masks.
    Supports consensus building across multiple annotators.
    """
    
    def __init__(self, data_root, bbox_shift=20, min_appearance=2):
        """
        Initialize dataset.
        
        Args:
            data_root (str): Root directory containing 'gts' and 'imgs' folders
            bbox_shift (int): Maximum random shift for bounding box augmentation
            min_appearance (int): Minimum appearances across annotators for consensus
        """
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, "gts")
        self.img_path = os.path.join(data_root, "imgs")
        self.bbox_shift = bbox_shift
        self.min_appearance = min_appearance
        
        # Find all ground truth files
        self.gt_path_files = sorted(
            glob.glob(os.path.join(self.gt_path, "**/*.npy"), recursive=True)
        )
        
        # Filter files that have corresponding images
        self.gt_path_files = [
            file for file in self.gt_path_files
            if os.path.isfile(os.path.join(self.img_path, os.path.basename(file)))
        ]
        
        print(f"Found {len(self.gt_path_files)} images in dataset")
    
    def __len__(self):
        return len(self.gt_path_files)
    
    def __getitem__(self, index):
        """
        Get a single data sample.
        
        Args:
            index (int): Sample index
            
        Returns:
            tuple: (image_tensor, gt_tensor, bbox_tensor, image_name)
        """
        # Load image
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            os.path.join(self.img_path, img_name), 
            "r", 
            allow_pickle=True
        )
        
        # Convert image to (C, H, W) format
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0, \
            "Image should be normalized to [0, 1]"
        
        # Load multi-annotator ground truth
        gt = np.load(self.gt_path_files[index], "r", allow_pickle=True)
        
        # Build consensus mask from multiple annotators
        gt2D_single_channel = self._build_consensus_mask(gt)
        
        # Verify mask values
        assert np.max(gt2D_single_channel) == 1 and np.min(gt2D_single_channel) == 0, \
            "Ground truth should contain only 0 and 1 values"
        
        # Generate bounding box from mask
        y_indices, x_indices = np.where(gt2D_single_channel > 0)
        
        if len(x_indices) == 0 or len(y_indices) == 0:
            # Handle empty masks by creating a small dummy box
            bboxes = np.array([0, 0, 10, 10])
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # Apply random augmentation to bounding box
            H, W = gt2D_single_channel.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            
            bboxes = np.array([x_min, y_min, x_max, y_max])
        
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D_single_channel[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )
    
    def _build_consensus_mask(self, gt_multi_annotator):
        """
        Build consensus mask from multiple annotator annotations.
        
        Args:
            gt_multi_annotator (numpy.ndarray): Multi-channel ground truth (C, H, W)
            
        Returns:
            numpy.ndarray: Single-channel consensus mask (H, W)
        """
        combined_gt = np.zeros_like(gt_multi_annotator[0], dtype=np.uint8)
        
        # Count valid annotations
        valid_annotations = 0
        for gt2D in gt_multi_annotator:
            unique_values = np.unique(gt2D)
            if np.array_equal(unique_values, [0, 1]) or np.array_equal(unique_values, [0]) or np.array_equal(unique_values, [1]):
                combined_gt += gt2D.astype(np.uint8)
                valid_annotations += 1
        
        if self.min_appearance > valid_annotations:
            # If min_appearance > number of annotations, use union of all masks
            gt2D_single_channel = np.where(combined_gt > 0, 1, 0)
        else:
            # Keep pixels that appear in at least min_appearance annotations
            gt2D_single_channel = np.where(combined_gt >= self.min_appearance, 1, 0)
        
        return gt2D_single_channel.astype(np.uint8)


class MedSAM(nn.Module):
    """
    MedSAM model wrapper around SAM components.
    Freezes the prompt encoder and trains image encoder + mask decoder.
    """
    
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        """
        Initialize MedSAM model.
        
        Args:
            image_encoder: SAM image encoder
            mask_decoder: SAM mask decoder
            prompt_encoder: SAM prompt encoder (will be frozen)
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        
        # Freeze prompt encoder parameters
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, image, box):
        """
        Forward pass through MedSAM.
        
        Args:
            image (torch.Tensor): Input image tensor (B, C, H, W)
            box (numpy.ndarray): Bounding box coordinates (B, 4)
            
        Returns:
            torch.Tensor: Predicted masks (B, 1, H, W)
        """
        # Encode image
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        
        # Process prompts (no gradients needed)
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)
            
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        
        # Decode masks
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        
        # Upscale to original resolution
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        
        return ori_res_masks


class IoULoss(nn.Module):
    """IoU (Intersection over Union) Loss for segmentation."""
    
    def __init__(self, reduction='mean'):
        super(IoULoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute IoU loss.
        
        Args:
            inputs (torch.Tensor): Predicted masks (logits)
            targets (torch.Tensor): Ground truth masks
            
        Returns:
            torch.Tensor: IoU loss
        """
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum(dim=1)
        union = (inputs + targets - inputs * targets).sum(dim=1)
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        loss = 1 - iou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs (torch.Tensor): Predicted masks (logits)
            targets (torch.Tensor): Ground truth masks
            
        Returns:
            torch.Tensor: Focal loss
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


def get_loss_function(loss_type, **kwargs):
    """
    Get loss function based on type.
    
    Args:
        loss_type (str): Type of loss function
        **kwargs: Additional arguments for loss function
        
    Returns:
        nn.Module: Loss function
    """
    if loss_type == "iou":
        return IoULoss(reduction="mean")
    elif loss_type == "focal":
        return FocalLoss(
            alpha=kwargs.get('focal_alpha', 0.25),
            gamma=kwargs.get('focal_gamma', 2.0),
            reduction="mean"
        )
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def visualize_batch(dataloader, save_path=None):
    """
    Visualize a batch of data for sanity checking.
    
    Args:
        dataloader (DataLoader): Data loader to visualize from
        save_path (str, optional): Path to save visualization
    """
    for step, (image, gt, bboxes, names_temp) in enumerate(dataloader):
        print(f"Batch shapes - Image: {image.shape}, GT: {gt.shape}, Boxes: {bboxes.shape}")
        
        _, axs = plt.subplots(1, 2, figsize=(15, 7))
        
        # Visualize two random samples from batch
        for i in range(min(2, image.shape[0])):
            idx = random.randint(0, image.shape[0] - 1) if image.shape[0] > 1 else 0
            
            # Show image with mask and bounding box
            axs[i].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
            show_mask(gt[idx].cpu().numpy(), axs[i])
            show_box(bboxes[idx].numpy(), axs[i])
            axs[i].axis("off")
            axs[i].set_title(f"Sample: {names_temp[idx]}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        break


def train_epoch(model, dataloader, optimizer, loss_fn, device, use_amp=False, scaler=None):
    """
    Train for one epoch.
    
    Args:
        model: MedSAM model
        dataloader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Training device
        use_amp (bool): Use automatic mixed precision
        scaler: GradScaler for AMP
        
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, (image, gt2D, boxes, _) in enumerate(progress_bar):
        optimizer.zero_grad()
        
        # Move data to device
        image = image.to(device)
        gt2D = gt2D.to(device)
        boxes_np = boxes.detach().cpu().numpy()
        
        if use_amp and scaler is not None:
            with autocast():
                pred_masks = model(image, boxes_np)
                loss = loss_fn(pred_masks, gt2D.float())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_masks = model(image, boxes_np)
            loss = loss_fn(pred_masks, gt2D.float())
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    return total_loss / num_batches


def validate_epoch(model, dataloader, loss_fn, device):
    """
    Validate for one epoch.
    
    Args:
        model: MedSAM model
        dataloader: Validation data loader
        loss_fn: Loss function
        device: Training device
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for step, (image, gt2D, boxes, _) in enumerate(progress_bar):
            # Move data to device
            image = image.to(device)
            gt2D = gt2D.to(device)
            boxes_np = boxes.detach().cpu().numpy()
            
            pred_masks = model(image, boxes_np)
            loss = loss_fn(pred_masks, gt2D.float())
            
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, loss, save_path, scaler=None):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch (int): Current epoch
        loss (float): Current loss
        save_path (str): Path to save checkpoint
        scaler: GradScaler for AMP (optional)
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    
    torch.save(checkpoint, save_path)


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create working directory
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = os.path.join(args.work_dir, f"{args.task_name}-{run_id}")
    os.makedirs(model_save_path, exist_ok=True)
    print(f"Model save path: {model_save_path}")
    
    # Initialize Weights & Biases if requested
    if args.use_wandb:
        try:
            import wandb
            if args.wandb_key:
                wandb.login(key=args.wandb_key)
            
            wandb.init(
                project=args.task_name,
                config=vars(args),
                name=f"{args.task_name}-{run_id}"
            )
        except ImportError:
            print("Warning: wandb not installed, skipping logging")
            args.use_wandb = False
    
    # Load SAM model
    print(f"Loading SAM model: {args.model_type}")
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    
    # Create MedSAM model
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in medsam_model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in medsam_model.parameters() if p.requires_grad):,}")
    
    # Set up optimizer
    trainable_params = list(medsam_model.image_encoder.parameters()) + \
                      list(medsam_model.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Set up loss function
    loss_fn = get_loss_function(
        args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )
    print(f"Using {args.loss_type} loss")
    
    # Set up AMP scaler
    scaler = GradScaler() if args.use_amp else None
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = NpyDataset(
        args.train_data_path,
        bbox_shift=args.bbox_shift,
        min_appearance=args.min_appearance
    )
    val_dataset = NpyDataset(
        args.val_data_path,
        bbox_shift=args.bbox_shift,
        min_appearance=args.min_appearance
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Visualize data if requested
    if args.visualize_data:
        print("Creating data visualization...")
        vis_path = os.path.join(model_save_path, "data_visualization.png")
        visualize_batch(train_dataloader, vis_path)
    
    # Resume training if checkpoint provided
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        medsam_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("loss", float('inf'))
        
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(
            medsam_model, train_dataloader, optimizer, loss_fn, 
            device, args.use_amp, scaler
        )
        
        # Validate
        val_loss = validate_epoch(medsam_model, val_dataloader, loss_fn, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
        
        # Save latest checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                medsam_model, optimizer, epoch, val_loss,
                os.path.join(model_save_path, "medsam_model_latest.pth"),
                scaler
            )
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                medsam_model, optimizer, epoch, val_loss,
                os.path.join(model_save_path, "medsam_model_best.pth"),
                scaler
            )
            print(f"New best model saved with validation loss: {best_loss:.4f}")
        
        # Plot training curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss", marker='o')
        plt.plot(val_losses, label="Validation Loss", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Curves - {args.loss_type.upper()} Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(model_save_path, "training_curves.png"), dpi=150)
        plt.close()
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Models saved in: {model_save_path}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()