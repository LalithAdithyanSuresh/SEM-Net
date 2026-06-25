#!/usr/bin/env python3
"""
Generate Validation Pairs for SEM-Net
======================================
Parses CMT evaluation metrics CSVs from the CCC folder to match image IDs with mask IDs.
Locates the original images and mask files on the system, resolves their paths,
and outputs .flist files for evaluation. Optionally copies the files to a target directory.

Usage:
  python generate_validation_pairs.py [options]

Example:
  python generate_validation_pairs.py --image-dir /mnt/datadrive/inpaint/places2/test_256 --mask-dir /mnt/datadrive/inpaint/iregularmask/test_mask/mask/testing_mask_dataset
"""

import os
import csv
import argparse
import shutil

def find_file(directory, filename, extensions=['.jpg', '.png', '.jpeg']):
    """Search for a file with various extensions in a directory."""
    if not os.path.exists(directory):
        return None
        
    base, _ = os.path.splitext(filename)
    for ext in extensions:
        candidate = os.path.join(directory, f"{base}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate validation flists/datasets from CMT evaluations in CCC.")
    parser.add_argument("--csv-dir", type=str, default=None,
                        help="Directory containing CCC evaluation metrics CSVs (e.g., metrics_SMALL.csv)")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory containing the original Places2/365 images")
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="Directory containing the original masks")
    parser.add_argument("--output-dir", type=str, default="./datasets/places365",
                        help="Directory where output .flist files will be saved")
    parser.add_argument("--copy-to", type=str, default=None,
                        help="Optional directory to physically copy the validation image-mask pairs to")
    args = parser.parse_args()

    # 1. Resolve CSV Directory
    csv_search_paths = [
        "CCC/CMT_Validate_Places2/results",
        "CCC/results",
        "CCC",
        "."
    ]
    csv_dir = args.csv_dir
    if not csv_dir:
        for p in csv_search_paths:
            if os.path.exists(p):
                csv_dir = p
                break
        if not csv_dir:
            csv_dir = "CCC/CMT_Validate_Places2/results"  # default fallback
    
    print(f"[*] Using CSV directory: {csv_dir}")

    # 2. Resolve Image Directory
    image_search_paths = [
        "datasets/places365/places365_standard/val",
        "/mnt/datadrive/inpaint/places2/test_256",
        "/home/snuc/Desktop/SEM-NETHybrid/TEMP_QUALITATIVE/places2/test_256",
        "."
    ]
    image_dir = args.image_dir
    if not image_dir:
        for p in image_search_paths:
            if os.path.exists(p):
                image_dir = p
                break
        if not image_dir:
            image_dir = "datasets/places365/places365_standard/val"  # default fallback
            
    print(f"[*] Using Image directory: {image_dir}")

    # 3. Resolve Mask Directory
    mask_search_paths = [
        "datasets/testing_mask_dataset",
        "/mnt/datadrive/inpaint/iregularmask/test_mask/mask/testing_mask_dataset",
        "/home/snuc/Desktop/SEM-NETHybrid/TEMP_QUALITATIVE/testing_mask_dataset",
        "."
    ]
    mask_dir = args.mask_dir
    if not mask_dir:
        for p in mask_search_paths:
            if os.path.exists(p):
                mask_dir = p
                break
        if not mask_dir:
            mask_dir = "datasets/testing_mask_dataset"  # default fallback
            
    print(f"[*] Using Mask directory: {mask_dir}")

    # 4. Process Categories
    categories = ["SMALL", "MEDIUM", "LARGE"]
    os.makedirs(args.output_dir, exist_ok=True)
    
    for cat in categories:
        csv_path = os.path.join(csv_dir, f"metrics_{cat}.csv")
        if not os.path.exists(csv_path):
            print(f"[!] Warning: Metrics CSV for {cat} not found at {csv_path}. Skipping category.")
            continue
            
        print(f"\nProcessing category: {cat}")
        img_paths = []
        mask_paths = []
        
        missing_images = 0
        missing_masks = 0
        
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                print(f"[!] CSV {csv_path} is empty. Skipping.")
                continue
                
            for row in reader:
                if not row or not row[0].strip() or row[0] == 'AVERAGE':
                    continue
                
                img_name = row[0]
                mask_id = row[1]
                
                # Try to locate the image
                found_img = find_file(image_dir, img_name)
                if not found_img:
                    missing_images += 1
                    # Log first few missing as samples
                    if missing_images <= 5:
                        print(f"  [?] Missing image file: '{img_name}' in {image_dir}")
                    # Use fallback path
                    found_img = os.path.join(image_dir, img_name)
                
                found_img_abs = os.path.abspath(found_img)
                    
                # Try to locate the mask (MaskID is e.g. 00000, file is 00000.png)
                mask_name = f"{mask_id}.png"
                found_mask = find_file(mask_dir, mask_name)
                if not found_mask:
                    missing_masks += 1
                    if missing_masks <= 5:
                        print(f"  [?] Missing mask file: '{mask_name}' in {mask_dir}")
                    found_mask = os.path.join(mask_dir, mask_name)
                
                found_mask_abs = os.path.abspath(found_mask)
                    
                img_paths.append(found_img_abs)
                mask_paths.append(found_mask_abs)
                
                # Optional physical copying
                if args.copy_to:
                    dest_img_dir = os.path.join(args.copy_to, "images", cat)
                    dest_mask_dir = os.path.join(args.copy_to, "masks", cat)
                    os.makedirs(dest_img_dir, exist_ok=True)
                    os.makedirs(dest_mask_dir, exist_ok=True)
                    
                    if os.path.exists(found_img_abs):
                        shutil.copy2(found_img_abs, os.path.join(dest_img_dir, os.path.basename(found_img_abs)))
                    if os.path.exists(found_mask_abs):
                        shutil.copy2(found_mask_abs, os.path.join(dest_mask_dir, os.path.basename(found_mask_abs)))

        # Write flists
        img_flist = os.path.join(args.output_dir, f"val_images_{cat}.flist")
        mask_flist = os.path.join(args.output_dir, f"val_masks_{cat}.flist")
        
        with open(img_flist, 'w', encoding='utf-8') as f_out:
            for p in img_paths:
                f_out.write(p + '\n')
                
        with open(mask_flist, 'w', encoding='utf-8') as f_out:
            for p in mask_paths:
                f_out.write(p + '\n')
                
        print(f"[+] Wrote {len(img_paths)} image paths to {img_flist}")
        print(f"[+] Wrote {len(mask_paths)} mask paths to {mask_flist}")
        if missing_images > 0:
            print(f"  [!] Warning: {missing_images} image files were not found locally.")
        if missing_masks > 0:
            print(f"  [!] Warning: {missing_masks} mask files were not found locally.")
        if args.copy_to:
            print(f"  [+] Physically copied validation pairs to: {args.copy_to}")

    print("\n[✓] Done! Flist files generated successfully.")

if __name__ == "__main__":
    main()
