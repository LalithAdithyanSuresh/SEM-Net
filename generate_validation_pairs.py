#!/usr/bin/env python3
"""
Generate Validation Pairs for SEM-Net
======================================
Parses CMT evaluation metrics CSVs from the CCC folder to match image IDs with mask IDs.
Locates the original images and mask files in the relative datasets directory, resolves their paths,
and outputs portable relative .flist files for evaluation.

Usage:
  python generate_validation_pairs.py [options]

Example:
  python generate_validation_pairs.py
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
    parser = argparse.ArgumentParser(description="Generate validation flists from CMT evaluations in CCC.")
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

    # 1. Resolve CSV/results Directory
    csv_search_paths = [
        "CMT_Validate_Places2/results",
        "results",
        "CCC/CMT_Validate_Places2/results",
        "CCC/results",
        "CCC",
        "."
    ]
    csv_dir = args.csv_dir
    if not csv_dir:
        for p in csv_search_paths:
            if os.path.exists(p) and os.path.isdir(p):
                has_small_dir = os.path.exists(os.path.join(p, "SMALL"))
                has_small_csv = os.path.exists(os.path.join(p, "metrics_SMALL.csv"))
                if has_small_dir or has_small_csv:
                    csv_dir = p
                    break
        if not csv_dir:
            csv_dir = "CMT_Validate_Places2/results"  # default fallback
    
    print(f"[*] Using CSV directory: {csv_dir}")

    # 2. Resolve Image Directory (relative dataset folder test_256)
    image_search_paths = [
        "datasets/places365/test_256",
        "datasets/places365/places365_standard/val",
        "/mnt/datadrive/inpaint/places2/test_256",
        "."
    ]
    image_dir = args.image_dir
    if not image_dir:
        for p in image_search_paths:
            if os.path.exists(p):
                image_dir = p
                break
        if not image_dir:
            image_dir = "datasets/places365/test_256"  # default fallback
            
    print(f"[*] Using Image directory: {image_dir}")

    # 3. Resolve Mask Directory
    mask_search_paths = [
        "datasets/testing_mask_dataset",
        "/mnt/datadrive/inpaint/iregularmask/test_mask/mask/testing_mask_dataset",
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
    output_dir_abs = os.path.abspath(args.output_dir)
    
    for cat in categories:
        # Determine if we can scan the folder directly
        cat_dir = os.path.join(csv_dir, cat)
        pairs = []
        
        if os.path.exists(cat_dir) and os.path.isdir(cat_dir):
            files = [f for f in os.listdir(cat_dir) if f.endswith('.png')]
            if files:
                print(f"[*] Found {len(files)} result images in folder: {cat_dir}. Parsing filenames...")
                files.sort()
                for f in files:
                    name_without_ext = os.path.splitext(f)[0]
                    parts = name_without_ext.split('_')
                    mask_idx = -1
                    for idx, part in enumerate(parts):
                        if part.isdigit() and len(part) == 5:
                            mask_idx = idx
                            break
                    if mask_idx != -1:
                        img_name = "_".join(parts[:mask_idx]) + ".jpg"
                        mask_id = parts[mask_idx]
                        pairs.append((img_name, mask_id))
        
        if not pairs:
            # Fallback to CSV
            csv_path = os.path.join(csv_dir, f"metrics_{cat}.csv")
            if not os.path.exists(csv_path):
                print(f"[!] Warning: Neither results folder nor CSV found for category {cat}. Skipping.")
                continue
                
            print(f"[*] Reading metrics from CSV: {csv_path}")
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
                    pairs.append((row[0], row[1]))
                    
        print(f"\nProcessing category: {cat} ({len(pairs)} pairs)")
        img_flist_lines = []
        mask_flist_lines = []
        
        missing_images = 0
        missing_masks = 0
        
        for img_name, mask_id in pairs:
            # Locate the image file
            found_img = find_file(image_dir, img_name)
            if not found_img:
                missing_images += 1
                if missing_images <= 5:
                    print(f"  [?] Missing image file: '{img_name}' in {image_dir}")
                found_img = os.path.join(image_dir, img_name)
            
            found_img_abs = os.path.abspath(found_img)
                
            # Locate the mask file (MaskID is e.g. 00000, file is 00000.png)
            mask_name = f"{mask_id}.png"
            found_mask = find_file(mask_dir, mask_name)
            if not found_mask:
                missing_masks += 1
                if missing_masks <= 5:
                    print(f"  [?] Missing mask file: '{mask_name}' in {mask_dir}")
                found_mask = os.path.join(mask_dir, mask_name)
            
            found_mask_abs = os.path.abspath(found_mask)
            
            # Compute relative paths from output_dir to the resolved files
            rel_img_path = os.path.relpath(found_img_abs, output_dir_abs).replace('\\', '/')
            rel_mask_path = os.path.relpath(found_mask_abs, output_dir_abs).replace('\\', '/')
            
            img_flist_lines.append(rel_img_path)
            mask_flist_lines.append(rel_mask_path)
            
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
            for line in img_flist_lines:
                f_out.write(line + '\n')
                
        with open(mask_flist, 'w', encoding='utf-8') as f_out:
            for line in mask_flist_lines:
                f_out.write(line + '\n')
                
        print(f"[+] Wrote {len(img_flist_lines)} image paths to {img_flist}")
        print(f"[+] Wrote {len(mask_flist_lines)} mask paths to {mask_flist}")
        if missing_images > 0:
            print(f"  [!] Warning: {missing_images} image files were not found locally.")
        if missing_masks > 0:
            print(f"  [!] Warning: {missing_masks} mask files were not found locally.")
        if args.copy_to:
            print(f"  [+] Physically copied validation pairs to: {args.copy_to}")

    print("\n[+] Done! Flist files generated successfully.")

if __name__ == "__main__":
    main()
