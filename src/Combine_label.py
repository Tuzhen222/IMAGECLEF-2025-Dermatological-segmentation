import os
import cv2
import numpy as np
from glob import glob
from scipy.stats import mode
import zipfile

# Configuration — easy to adjust
INPUT_FOLDER = "/kaggle/input/imageclefmed-mediqa-magic-2025/dermavqa-seg-trainvalid/dermavqa-segmentations/valid"
OUTPUT_FOLDER = "valid_labels"
PROGRESS_INTERVAL = 10  # report progress every N groups

def compress_folder_to_zip(folder_path, zip_name=None):
    """
    Compresses the given folder into a ZIP file alongside it.
    Returns the path to the created ZIP.
    """
    if zip_name is None:
        zip_name = f"{os.path.basename(folder_path)}.zip"
    zip_path = os.path.join(os.path.dirname(folder_path), zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for fname in files:
                file_path = os.path.join(root, fname)
                # Preserve relative folder structure inside the ZIP
                arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                zipf.write(file_path, arcname)
    return zip_path

def process_images():
    # Ensure output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Find all TIFF files in the input folder
    all_files = glob(os.path.join(INPUT_FOLDER, "*.tiff"))
    print(f"Total TIFF files found: {len(all_files)}")

    # Group files by their prefix before '_ann'
    file_groups = {}
    for path in all_files:
        base = os.path.basename(path).split("_ann")[0]
        file_groups.setdefault(base, []).append(path)
    print(f"Number of image groups to process: {len(file_groups)}")

    processed = 0
    for base, paths in file_groups.items():
        # Load all masks in this group
        images = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in paths]

        # Check that all images share the same shape
        shapes = [img.shape for img in images]
        if len(set(shapes)) != 1:
            print(f"⚠️ Inconsistent image sizes for group '{base}', skipping.")
            continue

        # Stack and perform per-pixel majority vote
        stack = np.stack(images, axis=0)
        merged, _ = mode(stack, axis=0, keepdims=False)
        merged = merged.astype(np.uint8)

        # Save the merged mask
        out_path = os.path.join(OUTPUT_FOLDER, f"{base}_mask_sys.tiff")
        cv2.imwrite(out_path, merged)
        processed += 1

        if processed % PROGRESS_INTERVAL == 0:
            print(f"✅ Processed {processed}/{len(file_groups)} groups")

    print(f"✅ Finished processing {processed}/{len(file_groups)} groups")

    # Compress the output folder to ZIP
    zip_path = compress_folder_to_zip(OUTPUT_FOLDER)
    print(f"✅ ZIP created at: {zip_path}")

if __name__ == "__main__":
    process_images()
