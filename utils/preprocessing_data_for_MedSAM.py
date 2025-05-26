import numpy as np
import os
import argparse
from skimage import transform
from tqdm import tqdm
import cv2  # For handling JPG, PNG images
import tifffile  # For handling TIFF files

def parse_arguments():
    """Parse command line arguments for the preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocess medical images and segmentation masks for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--img_path",
        type=str,
        required=True,
        help="Path to the directory containing training images"
    )
    
    parser.add_argument(
        "--gt_path", 
        type=str,
        required=True,
        help="Path to the directory containing ground truth segmentation masks"
    )
    
    parser.add_argument(
        "--output_path",
        type=str, 
        required=True,
        help="Path to save the processed .npy and .npz files"
    )
    
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Target size for resizing images (both width and height)"
    )
    
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (None for all images)"
    )
    
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Save individual .npy files for images and masks in addition to .npz files"
    )
    
    return parser.parse_args()

def load_image(img_path):
    """
    Load an image and convert it to RGB format.
    
    Args:
        img_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: RGB image array
    """
    # Load image as BGR (OpenCV default) and convert to RGB
    img_data = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_data is None:
        raise ValueError(f"Could not load image from {img_path}")
    
    # Convert from BGR to RGB
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    return img_data

def normalize_image(img_array):
    """
    Normalize image array to [0, 1] range.
    
    Args:
        img_array (numpy.ndarray): Input image array
        
    Returns:
        numpy.ndarray: Normalized image array
    """
    img_min = img_array.min()
    img_max = img_array.max()
    
    # Avoid division by zero by clipping the denominator
    img_normalized = (img_array - img_min) / np.clip(img_max - img_min, a_min=1e-8, a_max=None)
    return img_normalized

def resize_image(img_array, target_size, preserve_range=True):
    """
    Resize image to target size.
    
    Args:
        img_array (numpy.ndarray): Input image array
        target_size (int): Target size (width and height)
        preserve_range (bool): Whether to preserve the original value range
        
    Returns:
        numpy.ndarray: Resized image array
    """
    return transform.resize(
        img_array, 
        (target_size, target_size), 
        order=3,  # Bicubic interpolation for images
        preserve_range=preserve_range, 
        mode="constant", 
        anti_aliasing=True
    )

def resize_mask(mask_array, target_size):
    """
    Resize mask to target size using nearest neighbor interpolation.
    
    Args:
        mask_array (numpy.ndarray): Input mask array
        target_size (int): Target size (width and height)
        
    Returns:
        numpy.ndarray: Resized mask array
    """
    return transform.resize(
        mask_array, 
        (target_size, target_size), 
        order=0,  # Nearest neighbor interpolation for masks
        preserve_range=True, 
        mode="constant", 
        anti_aliasing=False
    )

def find_mask_files(gt_path, img_base_name):
    """
    Find all mask files corresponding to a given image.
    
    Args:
        gt_path (str): Path to ground truth directory
        img_base_name (str): Base name of the image (without extension)
        
    Returns:
        list: List of mask file names
    """
    all_files = os.listdir(gt_path)
    mask_files = [f for f in all_files if f.startswith(img_base_name)]
    return mask_files

def process_masks(mask_files, gt_path, target_size):
    """
    Process all mask files for a given image.
    
    Args:
        mask_files (list): List of mask file names
        gt_path (str): Path to ground truth directory
        target_size (int): Target size for resizing
        
    Returns:
        numpy.ndarray or None: Stacked mask tensor or None if no valid masks
    """
    valid_masks = []
    
    for mask_name in mask_files:
        # Extract annotator information from filename (e.g., ann0, ann1)
        annotator = mask_name.split('_')[-1].split('.')[0]
        
        # Load mask file (TIFF format)
        mask_file_path = os.path.join(gt_path, mask_name)
        try:
            mask_data = tifffile.imread(mask_file_path)
        except Exception as e:
            print(f"Warning: Could not load mask {mask_name}: {e}")
            continue
        
        # Resize mask to target size
        mask_resized = resize_mask(mask_data, target_size)
        
        # Only keep non-empty masks (masks with objects)
        if np.sum(mask_resized) > 0:
            valid_masks.append(np.uint8(mask_resized))
    
    # Stack all valid masks into a multi-channel tensor (C, H, W)
    if valid_masks:
        return np.stack(valid_masks, axis=0)
    else:
        return None

def save_processed_data(img_normalized, mask_tensor, img_base_name, output_path, save_individual=False):
    """
    Save processed image and mask data.
    
    Args:
        img_normalized (numpy.ndarray): Normalized image array
        mask_tensor (numpy.ndarray): Multi-channel mask tensor
        img_base_name (str): Base name of the image
        output_path (str): Output directory path
        save_individual (bool): Whether to save individual .npy files
    """
    # Always save as compressed .npz file containing both image and masks
    npz_path = os.path.join(output_path, f"{img_base_name}.npz")
    np.savez_compressed(npz_path, imgs=img_normalized, gts=mask_tensor)
    
    # Optionally save individual .npy files
    if save_individual:
        img_npy_path = os.path.join(output_path, "imgs", f"{img_base_name}.npy")
        mask_npy_path = os.path.join(output_path, "gts", f"{img_base_name}.npy")
        
        np.save(img_npy_path, img_normalized)
        np.save(mask_npy_path, mask_tensor)

def main():
    """Main preprocessing function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directories
    os.makedirs(args.output_path, exist_ok=True)
    if args.save_individual:
        os.makedirs(os.path.join(args.output_path, "imgs"), exist_ok=True)
        os.makedirs(os.path.join(args.output_path, "gts"), exist_ok=True)
    
    # Get list of image files in the input directory
    img_files = sorted(os.listdir(args.img_path))
    
    # Limit number of images if specified
    if args.max_images is not None:
        img_files = img_files[:args.max_images]
    
    print(f"Processing {len(img_files)} images...")
    
    processed_count = 0
    skipped_count = 0
    
    # Process each image file
    for img_name in tqdm(img_files, desc="Processing images"):
        try:
            # Get full path to image file
            img_file_path = os.path.join(args.img_path, img_name)
            
            # Skip if image file doesn't exist
            if not os.path.exists(img_file_path):
                skipped_count += 1
                continue
            
            # Load and process image
            img_data = load_image(img_file_path)
            img_resized = resize_image(img_data, args.image_size)
            img_normalized = normalize_image(img_resized)
            
            # Get base name without extension
            img_base_name = img_name.split('.')[0]
            
            # Find and process corresponding mask files
            mask_files = find_mask_files(args.gt_path, img_base_name)
            mask_tensor = process_masks(mask_files, args.gt_path, args.image_size)
            
            # Only save if we have valid masks
            if mask_tensor is not None:
                save_processed_data(
                    img_normalized, 
                    mask_tensor, 
                    img_base_name, 
                    args.output_path,
                    args.save_individual
                )
                processed_count += 1
            else:
                print(f"Warning: No valid masks found for image {img_name}")
                skipped_count += 1
                
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            skipped_count += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Skipped: {skipped_count} images")
    print(f"Output saved to: {args.output_path}")

if __name__ == "__main__":
    main()