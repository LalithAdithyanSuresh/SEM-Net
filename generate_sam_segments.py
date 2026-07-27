#!/usr/bin/env python3
"""
SAM Dataset Segment Map Generator
=================================
This script processes all images in a dataset directory (or specified via an flist file)
and generates per-pixel segmentation maps using Segment Anything Model (SAM / SAM 2 / FastSAM).

The output maps assign a unique segment ID to each pixel, which can be fed directly
into sequence models like Mamba to provide spatial segment awareness.

Usage:
------
1. Using Ultralytics SAM (default, fast):
   python generate_sam_segments.py --input_dir /path/to/dataset --output_dir /path/to/output_segments

2. Using Meta SAM (Official segment_anything repo):
   python generate_sam_segments.py --backend meta --input_dir /path/to/dataset --output_dir /path/to/output_segments --model_path sam_vit_b_01ec64.pth --model_type vit_b

3. Using an flist file:
   python generate_sam_segments.py --flist /path/to/val_images.flist --output_dir /path/to/output_segments
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch

def generate_color_palette(num_colors=500, seed=42):
    """Generates a deterministic color palette for visual segment rendering."""
    np.random.seed(seed)
    # Background (ID 0) is black [0, 0, 0]
    colors = np.random.randint(40, 255, size=(num_colors + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    return colors

def get_image_files(input_dir=None, flist=None):
    """Retrieves list of image file paths from directory or flist file."""
    image_paths = []
    if flist and os.path.exists(flist):
        with open(flist, 'r', encoding='utf-8') as f:
            for line in f:
                path = line.strip()
                if path:
                    image_paths.append(path)
    elif input_dir and os.path.exists(input_dir):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    image_paths.append(os.path.join(root, file))
    else:
        raise ValueError("Either --input_dir or --flist must be provided and exist!")
    
    return sorted(image_paths)

class UltralyticsSAMBackend:
    def __init__(self, model_name="sam_b.pt", device="cuda"):
        from ultralytics import SAM
        print(f"[SAM Backend] Initializing Ultralytics SAM model: {model_name} on {device}")
        self.model = SAM(model_name)
        self.device = device

    def segment_image(self, image_path, target_size=None):
        """Runs SAM inference and returns 2D segment ID map of shape (H, W)."""
        # Inference using ultralytics SAM
        results = self.model(image_path, device=self.device, verbose=False)
        
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        
        h, w = img.shape[:2]
        if target_size is not None and (target_size[0] != w or target_size[1] != h):
            h, w = target_size[1], target_size[0]

        segment_map = np.zeros((h, w), dtype=np.int32)

        if results and len(results) > 0 and results[0].masks is not None:
            masks_tensor = results[0].masks.data  # shape (N, H_orig, W_orig)
            masks = masks_tensor.cpu().numpy()
            
            # Sort masks by area descending so smaller details overlay on larger backgrounds
            areas = [np.sum(m) for m in masks]
            sorted_indices = np.argsort(areas)[::-1]
            
            for idx, mask_idx in enumerate(sorted_indices):
                m = masks[mask_idx]
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                
                segment_map[m > 0.5] = idx + 1  # 1-indexed segment IDs
                
        return segment_map, img

class MetaSAMBackend:
    def __init__(self, model_path, model_type="vit_b", device="cuda"):
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "segment_anything library not installed! Install via: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        
        print(f"[SAM Backend] Loading Meta SAM model: {model_type} from {model_path} on {device}")
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def segment_image(self, image_path, target_size=None):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None, None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(img_rgb)
        
        h, w = img_bgr.shape[:2]
        if target_size is not None and (target_size[0] != w or target_size[1] != h):
            h, w = target_size[1], target_size[0]

        segment_map = np.zeros((h, w), dtype=np.int32)
        
        # Sort masks by area descending
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        for idx, mask_dict in enumerate(masks):
            m = mask_dict['segmentation']
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            segment_map[m > 0] = idx + 1
            
        return segment_map, img_bgr

def process_dataset(args):
    # Select Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using Device: {device}")
    
    # Initialize SAM Backend
    if args.backend == "ultralytics":
        backend = UltralyticsSAMBackend(model_name=args.model_path, device=device)
    elif args.backend == "meta":
        backend = MetaSAMBackend(model_path=args.model_path, model_type=args.model_type, device=device)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # Gather images
    image_paths = get_image_files(input_dir=args.input_dir, flist=args.flist)
    print(f"Found {len(image_paths)} images to process.")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_color_map:
        color_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(color_dir, exist_ok=True)
    
    palette = generate_color_palette()

    # Pre-determine common base path for preserving folder structure
    base_dir = args.input_dir if args.input_dir else os.path.dirname(image_paths[0]) if image_paths else ""

    target_size = (args.width, args.height) if (args.width and args.height) else None

    success_count = 0

    for img_path in tqdm(image_paths, desc="Generating SAM Segments"):
        try:
            # Determine relative path structure
            if base_dir and img_path.startswith(base_dir):
                rel_path = os.path.relpath(img_path, base_dir)
            else:
                rel_path = os.path.basename(img_path)

            base_name, _ = os.path.splitext(rel_path)
            
            # Construct output image path
            out_img_path = os.path.join(args.output_dir, f"{base_name}_seg.png")
            os.makedirs(os.path.dirname(out_img_path), exist_ok=True)

            # Perform Segmentation
            segment_map, orig_img = backend.segment_image(img_path, target_size=target_size)

            if segment_map is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            max_id = segment_map.max()

            # Save Segment ID Map (Grayscale/Indexed Image for Mamba)
            # If max_id fits in uint8 (<= 255), save as 8-bit PNG; otherwise 16-bit PNG
            if max_id <= 255:
                seg_img = segment_map.astype(np.uint8)
            else:
                seg_img = segment_map.astype(np.uint16)
            
            cv2.imwrite(out_img_path, seg_img)

            # Optional: Save Color Visual Overlay
            if args.save_color_map:
                out_color_path = os.path.join(color_dir, f"{base_name}_color.png")
                os.makedirs(os.path.dirname(out_color_path), exist_ok=True)

                # Color mapping
                color_map = palette[np.clip(segment_map, 0, len(palette) - 1)]
                
                # Blend with original image if desired or save raw color map
                if orig_img is not None and orig_img.shape[:2] == color_map.shape[:2]:
                    color_bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
                    blended = cv2.addWeighted(orig_img, 0.5, color_bgr, 0.5, 0)
                    cv2.imwrite(out_color_path, blended)
                else:
                    cv2.imwrite(out_color_path, cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))

            success_count += 1

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")

    print(f"\nCompleted! Successfully processed {success_count}/{len(image_paths)} images.")
    print(f"Segmentation maps saved to: {os.path.abspath(args.output_dir)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate SAM Segmentation Maps for Dataset")
    parser.add_argument("--input_dir", type=str, default=None, help="Path to input dataset image directory")
    parser.add_argument("--flist", type=str, default=None, help="Path to flist file containing list of image paths")
    parser.add_argument("--output_dir", type=str, default="./sam_segmentations", help="Directory to save segment maps")
    
    parser.add_argument("--backend", type=str, choices=["ultralytics", "meta"], default="ultralytics",
                        help="SAM backend library to use (ultralytics or meta)")
    parser.add_argument("--model_path", type=str, default="sam_b.pt",
                        help="Path or weight name for SAM model (e.g., sam_b.pt, sam2.1_b.pt, sam_vit_h_4b8939.pth)")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"],
                        help="Meta SAM model architecture type (only used if backend=meta)")
    
    parser.add_argument("--width", type=int, default=None, help="Target width to resize segmentation maps (optional)")
    parser.add_argument("--height", type=int, default=None, help="Target height to resize segmentation maps (optional)")
    parser.add_argument("--device", type=str, default="auto", help="Device to run inference ('auto', 'cuda', 'cpu')")
    parser.add_argument("--save_color_map", action="store_true", help="Save visual RGB colorized segment images for debugging")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_dataset(args)
