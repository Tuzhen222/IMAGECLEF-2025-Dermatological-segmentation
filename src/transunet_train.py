import os
import sys
import subprocess
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
import torch.nn as nn
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import glob

def setup_environment():
    """Install required packages, clone repos, download and unzip pretrained models."""
    # Install core dependencies
    subprocess.run(['pip', 'install', 'segmentation_models_pytorch', 'imagecodecs'], check=True)
    subprocess.run(['pip', 'install', 'medpy', 'ml_collections', 'gdown'], check=True)

    # Clone TransUNet if needed
    if not os.path.isdir('TransUNet'):
        print("Cloning TransUNet repository...")
        subprocess.run(['git', 'clone', 'https://github.com/Beckschen/TransUNet.git'], check=True)

    # Install TransUNet requirements
    req_file = os.path.join('TransUNet', 'requirements.txt')
    if os.path.exists(req_file):
        subprocess.run(['pip', 'install', '-r', req_file], check=True)

    # Download pretrained models archive via gdown
    zip_url = "https://drive.google.com/file/d/1mGhBIDcyollAEz0JMafXUO5zO3hErq4H"
    print("Downloading pretrained models...")
    subprocess.run(['gdown', zip_url, '--fuzzy'], check=True)

    # Unzip pretrained_models.zip into TransUNet/pretrained_models
    zip_candidates = glob.glob('*.zip')
    for z in zip_candidates:
        if 'pretrained_models' in z:
            dst = os.path.join('TransUNet', 'pretrained_models')
            os.makedirs(dst, exist_ok=True)
            subprocess.run(['unzip', '-o', z, '-d', dst], check=True)
            break

class CustomSegmentationDataset(Dataset):
    """Custom dataset for segmentation training and validation."""
    def __init__(self, image_dirs, mask_dirs, transform=None):
        self.transform = transform
        # gather image paths
        self.image_paths = []
        for d in image_dirs:
            self.image_paths += sorted(Path(d).glob("*.jpg")) + sorted(Path(d).glob("*.png"))
        # map mask stems to mask paths
        self.mask_map = {}
        for d in mask_dirs:
            for m in Path(d).glob("*.tiff"):
                self.mask_map[m.stem] = m
        # keep only images with masks
        self.image_paths = [p for p in self.image_paths if p.stem in self.mask_map]
        print(f"Found {len(self.image_paths)} images with corresponding masks.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_map[img_path.stem]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype(np.uint8)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            img_t = aug['image']
            mask_t = aug['mask'].unsqueeze(0).float()
        else:
            default_tf = A.Compose([
                A.Resize(224,224),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2(),
            ])
            aug = default_tf(image=image, mask=mask)
            img_t = aug['image']
            mask_t = aug['mask'].unsqueeze(0).float()

        return img_t, mask_t

def get_transforms():
    """Return training and validation augmentations."""
    train_tf = A.Compose([
        A.Resize(224,224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
    return train_tf, val_tf

def load_transunet_model(vit_name='R50-ViT-B_16',
                         img_size=224,
                         n_classes=1,
                         vit_patches_size=16,
                         pretrained_dir='TransUNet/pretrained_models'):
    """Load TransUNet architecture and optional ImageNet weights."""
    sys.path.append('TransUNet')
    from networks.vit_seg_modeling import VisionTransformer as ViT_seg
    from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

    cfg = CONFIGS_ViT_seg[vit_name]
    cfg.n_classes = n_classes
    if 'R50' in vit_name:
        cfg.n_skip = 3
        cfg.patches.grid = (img_size//vit_patches_size, img_size//vit_patches_size)
    else:
        cfg.n_skip = 0

    model = ViT_seg(cfg, img_size=img_size, num_classes=n_classes)

    mapping = {
        'R50-ViT-B_16': 'imagenet21k_R50+ViT-B_16.npz',
        'ViT-B_16':      'imagenet21k_ViT-B_16.npz',
        'ViT-B_32':      'imagenet21k_ViT-B_32.npz',
        'ViT-L_32':      'imagenet21k_ViT-L_32.npz',
        'ViT-L_16':      'imagenet21k_ViT-L_16.npz',
    }
    if vit_name in mapping:
        wpath = os.path.join(pretrained_dir, mapping[vit_name])
        if os.path.exists(wpath):
            weights = np.load(wpath)
            model.load_from(weights)
            print(f"Loaded pretrained weights from {wpath}")
        else:
            print(f"[Warning] Weights not found at {wpath}, training from scratch.")
    return model

class TransUNetWrapper(nn.Module):
    """Wrap TransUNet to upsample outputs to target size."""
    def __init__(self, base_model, target_size=(224,224)):
        super().__init__()
        self.base_model = base_model
        self.target_size = target_size

    def forward(self, x):
        out = self.base_model(x)
        if out.shape[2:] != self.target_size:
            out = F.interpolate(out, size=self.target_size,
                                mode='bilinear', align_corners=True)
        return out

def dice_coeff(pred, tgt):
    """Compute Dice coefficient."""
    smooth = 1e-5
    p = torch.sigmoid(pred) > 0.5
    inter = (p.float() * tgt).sum()
    return (2.*inter + smooth) / (p.sum() + tgt.sum() + smooth)

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, total_dice = 0, 0
    for imgs, masks in tqdm(loader, desc="Train"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outs = model(imgs)
            loss = criterion(outs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_dice += dice_coeff(outs, masks).item()
    n = len(loader)
    return total_loss/n, total_dice/n

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice = 0, 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validate"):
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.cuda.amp.autocast():
                outs = model(imgs)
                loss = criterion(outs, masks)
            total_loss += loss.item()
            total_dice += dice_coeff(outs, masks).item()
    n = len(loader)
    return total_loss/n, total_dice/n

def plot_history(hist):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(hist['train_loss'], label='Train Loss')
    plt.plot(hist['val_loss'],   label='Val Loss')
    plt.legend(); plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(hist['train_dice'], label='Train Dice')
    plt.plot(hist['val_dice'],   label='Val Dice')
    plt.legend(); plt.title("Dice")
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

def visualize_preds(model, loader, device, n=3):
    model.eval()
    imgs_list, masks_list, preds_list = [], [], []
    with torch.no_grad():
        for imgs, masks in loader:
            if len(imgs_list) >= n: break
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.cuda.amp.autocast():
                outs = torch.sigmoid(model(imgs)) > 0.5
            for i in range(imgs.size(0)):
                if len(imgs_list) >= n: break
                img_np = imgs.cpu()[i].permute(1,2,0).numpy()
                img_np = img_np * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
                imgs_list.append(img_np.clip(0,1))
                masks_list.append(masks.cpu()[i,0].numpy())
                preds_list.append(outs.cpu()[i,0].numpy())
    fig, axs = plt.subplots(n, 3, figsize=(12,4*n))
    for i in range(n):
        axs[i,0].imshow(imgs_list[i]);         axs[i,0].axis('off'); axs[i,0].set_title("Image")
        axs[i,1].imshow(masks_list[i], cmap='gray'); axs[i,1].axis('off'); axs[i,1].set_title("Truth")
        axs[i,2].imshow(preds_list[i], cmap='gray'); axs[i,2].axis('off'); axs[i,2].set_title("Pred")
    plt.tight_layout()
    plt.savefig("predictions.png")
    plt.show()

def main():
    setup_environment()

    # data paths
    augment_folders = [
        "AdditiveNoise","AdvancedAugmentation","AdvancedAugmentation_2",
        "CenterCrop","CompositeAug","ElasticTransform","GridDistortion",
        "HorizontalFlip","Medium","Medium_add_non_spatial_transformations",
        "OpticalDistortion","Original","PadIfNeeded","RandomRotate90",
        "RandomSizedCrop","RandomSizedCrop_0.1","Transpose","VerticalFlip",
        "gpt1","gpt2","gpt3"
    ]
    img_base = "/kaggle/input/datasetimageclef-augmentation-final/images_train"
    lbl_base = "/kaggle/input/datasetimageclef-augmentation-final/labels_train"
    imgs_train = [f"{img_base}/{d}" for d in augment_folders]
    lbls_train = [f"{lbl_base}/{d}" for d in augment_folders]
    imgs_val   = ["/kaggle/input/datasetimageclef-augmentation-final/images_valid"]
    lbls_val   = ["/kaggle/input/datasetimageclef-augmentation-final/labels_valid"]

    train_tf, val_tf = get_transforms()
    train_ds = CustomSegmentationDataset(imgs_train, lbls_train, transform=train_tf)
    val_ds   = CustomSegmentationDataset(imgs_val,   lbls_val,   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransUNetWrapper(
        load_transunet_model(vit_name='R50-ViT-B_16'),
        target_size=(224,224)
    ).to(device)

    epochs = 7
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    criterion = smp.losses.DiceLoss(mode='binary')
    scaler    = torch.cuda.amp.GradScaler()

    best_loss = float('inf')
    history = {'train_loss':[], 'val_loss':[], 'train_dice':[], 'val_dice':[]}

    for ep in range(epochs):
        print(f"Epoch {ep+1}/{epochs}")
        tl, td = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        vl, vd = validate_epoch(model, val_loader, optimizer, device)
        history['train_loss'].append(tl); history['train_dice'].append(td)
        history['val_loss'].append(vl);   history['val_dice'].append(vd)
        print(f"  Train loss: {tl:.4f}, dice: {td:.4f}")
        print(f"  Val   loss: {vl:.4f}, dice: {vd:.4f}")
        scheduler.step(vl)
        if vl < best_loss:
            best_loss = vl
            torch.save(model.state_dict(), "TransUNet_best_model.pth")
            print("  Saved best model.")
    print("Training complete. Best val loss:", best_loss)

    plot_history(history)
    visualize_preds(model, val_loader, device, n=3)

if __name__ == "__main__":
    main()
