import os
import sys
import subprocess
import json

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
import tifffile
import segmentation_models_pytorch as smp

def setup_environment():
    """Install required packages and clone necessary repositories."""
    print("==> Installing Python packages...")
    subprocess.run(['pip', 'install', 'segmentation_models_pytorch'], check=True)
    subprocess.run(['pip', 'install', 'imagecodecs'], check=True)
    subprocess.run(['pip', 'install', 'medpy', 'ml_collections'], check=True)
    # TransUNet repo
    if not os.path.isdir('TransUNet'):
        print("==> Cloning TransUNet repository...")
        subprocess.run(
            ['git', 'clone', 'https://github.com/Beckschen/TransUNet.git'],
            check=True
        )
    # Install TransUNet requirements
    req_file = os.path.join('TransUNet', 'requirements.txt')
    if os.path.exists(req_file):
        print("==> Installing TransUNet requirements...")
        subprocess.run(['pip', 'install', '-r', req_file], check=True)
    else:
        print(f"[Warning] {req_file} not found, skipping.")
    # ImageCLEF-MAGIC-2025 repo for scoring
    if not os.path.isdir('ImageCLEF-MAGIC-2025'):
        print("==> Cloning ImageCLEF-MAGIC-2025 repository...")
        subprocess.run(
            ['git', 'clone', 'https://github.com/wyim/ImageCLEF-MAGIC-2025.git'],
            check=True
        )

class PredictDataset(Dataset):
    """Dataset for loading images for prediction."""
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(
            list(self.image_dir.glob("*.jpg")) +
            list(self.image_dir.glob("*.png"))
        )
        self.transform = transform
        print(f"Found {len(self.image_paths)} images for prediction.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        image_np = np.array(image)

        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            default_tf = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485,0.456,0.406),
                            std=(0.229,0.224,0.225)),
                ToTensorV2(),
            ])
            augmented = default_tf(image=image_np)
            image_tensor = augmented['image']

        return image_tensor, str(img_path), width, height

def custom_collate_fn(batch):
    """Collate fn to keep original sizes alongside tensors."""
    images  = torch.stack([item[0] for item in batch])
    paths   = [item[1] for item in batch]
    widths  = [item[2] for item in batch]
    heights = [item[3] for item in batch]
    return images, paths, widths, heights

def get_predict_transform():
    """Albumentations transform for prediction."""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485,0.456,0.406),
                    std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

def load_transunet_model(vit_name='R50-ViT-B_16',
                         img_size=224,
                         n_classes=1,
                         vit_patches_size=16,
                         pretrained_dir='pretrained_models',
                         load_pretrained=False):
    """Load TransUNet architecture, optionally with pretrained weights."""
    # Ensure the TransUNet repo is in path
    sys.path.append(os.path.abspath('TransUNet'))
    from networks.vit_seg_modeling import VisionTransformer as ViT_seg
    from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

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
        mapping = {
            'R50-ViT-B_16': 'imagenet21k_R50+ViT-B_16.npz',
            'ViT-B_16':      'imagenet21k_ViT-B_16.npz',
            'ViT-B_32':      'imagenet21k_ViT-B_32.npz',
            'ViT-L_32':      'imagenet21k_ViT-L_32.npz',
            'ViT-L_16':      'imagenet21k_ViT-L_16.npz'
        }
        if vit_name not in mapping:
            raise ValueError(f"Unsupported vit_name '{vit_name}'.")
        weights_file = os.path.join(pretrained_dir, mapping[vit_name])
        if os.path.exists(weights_file):
            weights = np.load(weights_file)
            model.load_from(weights)
            print(f"Loaded pretrained weights from {weights_file}")
        else:
            print(f"[Warning] Pretrained file not found at {weights_file}")

    return model

class TransUNetWrapper(nn.Module):
    """Wrap TransUNet to upsample output to target size."""
    def __init__(self, base_model, target_size=(224,224)):
        super().__init__()
        self.base_model = base_model
        self.target_size = target_size

    def forward(self, x):
        out = self.base_model(x)
        if out.shape[2:] != self.target_size:
            out = F.interpolate(
                out, size=self.target_size,
                mode='bilinear', align_corners=True
            )
        return out

def predict_all(model, loader, device, save_dir='predictions'):
    """Run inference on all images and save masks as TIFF."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    print("Starting prediction...")
    with torch.no_grad():
        for batch_idx, (images, img_paths, widths, heights) in enumerate(
            tqdm(loader, desc="Predicting")
        ):
            images = images.to(device)
            outputs = model(images)
            if outputs.shape[2:] != (224,224):
                outputs = F.interpolate(
                    outputs, size=(224,224),
                    mode='bilinear', align_corners=True
                )
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()  # (B,1,224,224)

            for i in range(preds.shape[0]):
                if batch_idx < 2:
                    print(f"[Batch {batch_idx}] {img_paths[i]} -> resizing back to ({widths[i]}, {heights[i]})")
                mask = preds[i,0].astype(np.uint8)
                mask_img = Image.fromarray(mask)
                mask_resized = mask_img.resize((widths[i], heights[i]), resample=Image.NEAREST)
                out_path = os.path.join(save_dir, f"{Path(img_paths[i]).stem}_mask_sys.tiff")
                tifffile.imwrite(out_path, np.array(mask_resized))
                if batch_idx < 2:
                    print(f"Saved: {out_path}")

    print(f"All predictions saved to '{save_dir}'")

def main_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_folder = "/kaggle/input/imageclefmed-mediqa-magic-2025/images_final/images_final/images_test"
    transform = get_predict_transform()
    dataset = PredictDataset(input_folder, transform=transform)
    loader  = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

    base_model = load_transunet_model(
        vit_name='R50-ViT-B_16', img_size=224,
        n_classes=1, vit_patches_size=16,
        pretrained_dir='pretrained_models',
        load_pretrained=False
    )
    model = TransUNetWrapper(base_model, target_size=(224,224)).to(device)

    ckpt = "/kaggle/input/transunetne/TransUNet_best_model_R50+ViT-B_16_1.pth"
    if os.path.exists(ckpt):
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        except TypeError:
            model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded checkpoint '{ckpt}'")
    else:
        print(f"[Error] Checkpoint not found at '{ckpt}'")
        return

    predict_all(model, loader, device, save_dir='TransUnet_R50-ViT-B_16')

def evaluate_scores():
    """Run segmentation scoring and print JSON results."""
    # Install scoring deps
    subprocess.run(['pip', 'install', 'imagecodecs', 'gdown'], check=True)
    eval_dir = os.path.join('ImageCLEF-MAGIC-2025', 'evaluation')
    if not os.path.isdir(eval_dir):
        print(f"[Error] Evaluation folder not found: {eval_dir}")
        return

    labels_dir = "/kaggle/input/datasetimageclef-augmentation-final/labels_test"
    results_dir = "/kaggle/working/swin_unet_result"
    out_dir = "/kaggle/working"

    cmd = [
        sys.executable, "score_segmentations.py",
        labels_dir, results_dir, out_dir, "sys"
    ]
    print(f"Running scoring: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=eval_dir, check=True)

    scores_file = os.path.join(out_dir, "scores_segmentation.json")
    if os.path.exists(scores_file):
        with open(scores_file, "r") as f:
            data = json.load(f)
        print(json.dumps(data, indent=4, ensure_ascii=False))
    else:
        print(f"[Error] Scores file not found: {scores_file}")

if __name__ == "__main__":
    setup_environment()
    main_predict()
    evaluate_scores()
