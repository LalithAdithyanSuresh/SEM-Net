#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def process_single_image(img_path, output_dir, model, device, lock):
    filename = os.path.basename(img_path)
    base_name, _ = os.path.splitext(filename)
    out_path = os.path.join(output_dir, f"{base_name}.png")
    
    if os.path.exists(out_path):
        return True

    try:
        with lock:
            results = model(img_path, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, verbose=False)
        
        if results and len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            if len(masks) > 0:
                combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
            else:
                img = cv2.imread(img_path)
                h, w = img.shape[:2] if img is not None else (256, 256)
                combined_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            img = cv2.imread(img_path)
            h, w = img.shape[:2] if img is not None else (256, 256)
            combined_mask = np.zeros((h, w), dtype=np.uint8)

        cv2.imwrite(out_path, combined_mask)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def generate_masks_for_dir(input_dir, output_dir, num_workers=8, model_name="FastSAM-s.pt"):
    if not os.path.exists(input_dir):
        print(f"[FastSAM] Input directory {input_dir} does not exist. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[FastSAM] Loading model {model_name}...")
    try:
        from ultralytics import FastSAM
    except ImportError:
        raise RuntimeError("The 'ultralytics' package is required for FastSAM segment mask generation. Install it via 'pip install ultralytics'.")

    model = FastSAM(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    image_paths = sorted(list(set(image_paths)))
    if not image_paths:
        print(f"[FastSAM] No images found in {input_dir}.")
        return

    print(f"[FastSAM] Processing {len(image_paths)} images from {input_dir} -> {output_dir} using {num_workers} workers on device '{device}'...")

    lock = threading.Lock()
    failed_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_image, img_p, output_dir, model, device, lock): img_p
            for img_p in image_paths
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Segmenting {os.path.basename(output_dir)}"):
            res = future.result()
            if not res:
                failed_count += 1

    print(f"[FastSAM] Completed segment mask generation for {output_dir}. (Failed: {failed_count})")

def main():
    parser = argparse.ArgumentParser(description="Generate segment masks using FastSAM with parallel workers.")
    parser.add_argument("--dataset-dir", default="dataset", help="Root dataset directory containing train/test folders")
    parser.add_argument("--train-dir", default=None, help="Train image folder path")
    parser.add_argument("--test-dir", default=None, help="Test image folder path")
    parser.add_argument("--model", default="FastSAM-s.pt", help="FastSAM model weights file")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker threads (default: 8)")
    args = parser.parse_args()

    # Determine train input and output
    train_input = args.train_dir or os.path.join(args.dataset_dir, "train")
    if not os.path.exists(train_input) and os.path.exists("train"):
        train_input = "train"
    train_output = os.path.join(os.path.dirname(train_input) if os.path.dirname(train_input) else ".", "train_seg")

    # Determine test input and output
    test_input = args.test_dir or os.path.join(args.dataset_dir, "test")
    if not os.path.exists(test_input) and os.path.exists("test"):
        test_input = "test"
    test_output = os.path.join(os.path.dirname(test_input) if os.path.dirname(test_input) else ".", "test_seg")

    print(f"[FastSAM Segment Generator]")
    print(f"  Train: {train_input} -> {train_output}")
    print(f"  Test:  {test_input} -> {test_output}")
    print(f"  Workers: {args.workers}")

    if os.path.exists(train_input):
        generate_masks_for_dir(train_input, train_output, num_workers=args.workers, model_name=args.model)
    
    if os.path.exists(test_input):
        generate_masks_for_dir(test_input, test_output, num_workers=args.workers, model_name=args.model)

if __name__ == "__main__":
    main()
