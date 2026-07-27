#!/usr/bin/env python3
"""
Ultra-Fast Multi-GPU SAM & FastSAM Dataset Segment Map Generator
=================================================================
This script processes all images in a dataset directory (or via an flist file)
and generates per-pixel segmentation maps using Segment Anything Model (SAM / SAM 2 / FastSAM / MobileSAM).

Supported Models:
-----------------
- FastSAM: 'fastsam-s.pt', 'fastsam-x.pt' (Ultra-fast CNN based segmentation)
- SAM 2 / 2.1: 'sam2.1_t.pt', 'sam2.1_s.pt', 'sam2.1_b.pt', 'sam2.1_l.pt' (Next-Gen SAM 2)
- MobileSAM: 'mobile_sam.pt' (Lightweight SAM)
- SAM 1: 'sam_b.pt', 'sam_l.pt', 'sam_h.pt'

Usage (Inside LAST directory):
------------------------------
# Ultra-fast FastSAM (Recommended for massive datasets):
python generate_sam_segments.py --input_dir Dataset/train --output_dir Dataset/train_map --model_path fastsam-s.pt --gpus 0 1 --workers_per_gpu 8

# SAM 2.1 Tiny:
python generate_sam_segments.py --input_dir Dataset/train --output_dir Dataset/train_map --model_path sam2.1_t.pt --gpus 0 1 --workers_per_gpu 8
"""

import os
import sys
import tempfile

# Silence Matplotlib & Ultralytics warnings
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
os.environ["YOLO_VERBOSE"] = "False"

import glob
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import torch.multiprocessing as mp

def generate_color_palette(num_colors=1000, seed=42):
    """Generates a deterministic color palette for visual segment rendering."""
    np.random.seed(seed)
    colors = np.random.randint(40, 255, size=(num_colors + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background (ID 0) is black
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

def save_segment_map_task(task_args):
    """Async task for writing PNG segment maps to disk on CPU thread pool."""
    out_img_path, seg_img, out_color_path, color_map, orig_img = task_args
    try:
        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        cv2.imwrite(out_img_path, seg_img)

        if out_color_path and color_map is not None:
            os.makedirs(os.path.dirname(out_color_path), exist_ok=True)
            if orig_img is not None and orig_img.shape[:2] == color_map.shape[:2]:
                color_bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
                blended = cv2.addWeighted(orig_img, 0.5, color_bgr, 0.5, 0)
                cv2.imwrite(out_color_path, blended)
            else:
                cv2.imwrite(out_color_path, cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Error saving {out_img_path}: {e}")

def process_single_result(res, img, img_path, base_dir, args, palette):
    """Extracts segment map from single SAM/FastSAM result."""
    h, w = img.shape[:2]
    target_size = (args.width, args.height) if (args.width and args.height) else None
    if target_size is not None and (target_size[0] != w or target_size[1] != h):
        out_h, out_w = target_size[1], target_size[0]
    else:
        out_h, out_w = h, w

    segment_map = np.zeros((out_h, out_w), dtype=np.int32)

    if res is not None and res.masks is not None:
        masks_tensor = res.masks.data  # shape (N, H, W)
        masks = masks_tensor.cpu().numpy()

        # Sort by mask area descending
        areas = [np.sum(m) for m in masks]
        sorted_indices = np.argsort(areas)[::-1]

        for idx, mask_idx in enumerate(sorted_indices):
            m = masks[mask_idx]
            if m.shape[:2] != (out_h, out_w):
                m = cv2.resize(m.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            segment_map[m > 0.5] = idx + 1  # 1-indexed segment IDs

    # Determine output relative path
    if base_dir and img_path.startswith(base_dir):
        rel_path = os.path.relpath(img_path, base_dir)
    else:
        rel_path = os.path.basename(img_path)

    base_name, _ = os.path.splitext(rel_path)
    out_img_path = os.path.join(args.output_dir, f"{base_name}_seg.png")

    max_id = segment_map.max()
    if max_id <= 255:
        seg_img = segment_map.astype(np.uint8)
    else:
        seg_img = segment_map.astype(np.uint16)

    out_color_path = None
    color_map = None
    if args.save_color_map:
        color_dir = os.path.join(args.output_dir, "visualizations")
        out_color_path = os.path.join(color_dir, f"{base_name}_color.png")
        color_map = palette[np.clip(segment_map, 0, len(palette) - 1)]

    return (out_img_path, seg_img, out_color_path, color_map, img)

def load_sam_model(args, device):
    """Loads appropriate SAM / SAM 2 / FastSAM model based on model_path."""
    model_name_lower = args.model_path.lower()

    if args.backend == "ultralytics":
        if "fastsam" in model_name_lower:
            from ultralytics import FastSAM
            # Format model name (e.g. FastSAM-s.pt)
            ckpt = "FastSAM-s.pt" if "s" in model_name_lower else "FastSAM-x.pt" if "x" in model_name_lower else args.model_path
            print(f"[Worker] Loading FastSAM model: {ckpt} on {device}")
            return FastSAM(ckpt)
        else:
            from ultralytics import SAM
            print(f"[Worker] Loading SAM model: {args.model_path} on {device}")
            return SAM(args.model_path)
    else:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry[args.model_type](checkpoint=args.model_path)
        sam.to(device=device)
        return SamAutomaticMaskGenerator(sam)

def gpu_worker_process(gpu_id, image_paths, base_dir, args, progress_queue):
    """Worker process executed per GPU stream."""
    if gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    # Initialize SAM / FastSAM model for this worker
    model = load_sam_model(args, device)
    palette = generate_color_palette()
    save_pool = ThreadPoolExecutor(max_workers=args.num_save_workers)

    is_fastsam = "fastsam" in args.model_path.lower()

    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            progress_queue.put(1)
            continue

        try:
            with torch.cuda.amp.autocast(enabled=(args.fp16 and gpu_id >= 0)):
                if args.backend == "ultralytics":
                    if is_fastsam:
                        # FastSAM inference with texts/retina_masks options
                        results = model(img, device=device, retina_masks=True, verbose=False)
                    else:
                        results = model(img, device=device, verbose=False)
                    res = results[0] if results else None
                else:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    res = model.generate(img_rgb)

            task = process_single_result(res, img, p, base_dir, args, palette)
            save_pool.submit(save_segment_map_task, task)
        except Exception as e:
            print(f"\nError processing {p}: {e}")

        progress_queue.put(1)

    save_pool.shutdown(wait=True)

def main():
    args = parse_args()

    if args.gpus == ["auto"]:
        if torch.cuda.is_available():
            gpus = list(range(torch.cuda.device_count()))
        else:
            gpus = [-1]
    else:
        gpus = [int(g) for g in args.gpus]

    print(f"[SAM Parallel] Target GPUs: {gpus} | Model: {args.model_path} | Workers/GPU: {args.workers_per_gpu}")

    image_paths = get_image_files(input_dir=args.input_dir, flist=args.flist)
    total_images = len(image_paths)
    print(f"[SAM Parallel] Found {total_images} total images to process.")

    if total_images == 0:
        print("No images found. Exiting.")
        return

    base_dir = args.input_dir if args.input_dir else os.path.dirname(image_paths[0])

    if len(gpus) == 1 and gpus[0] == -1:
        print("Running in CPU single-process mode...")
        progress_queue = mp.Queue()
        gpu_worker_process(-1, image_paths, base_dir, args, progress_queue)
        return

    worker_list = []
    for gpu_id in gpus:
        for _ in range(args.workers_per_gpu):
            worker_list.append(gpu_id)

    total_workers = len(worker_list)
    chunks = [image_paths[i::total_workers] for i in range(total_workers)]

    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    processes = []

    start_time = time.time()

    for idx, gpu_id in enumerate(worker_list):
        p = ctx.Process(
            target=gpu_worker_process,
            args=(gpu_id, chunks[idx], base_dir, args, progress_queue)
        )
        p.start()
        processes.append(p)

    with tqdm(total=total_images, desc="Multi-GPU SAM Generation") as pbar:
        completed = 0
        while completed < total_images:
            try:
                count = progress_queue.get(timeout=1.0)
                completed += count
                pbar.update(count)
            except Exception:
                if all(not p.is_alive() for p in processes):
                    break

    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"\n[SAM Parallel] Done! Processed {total_images} images in {elapsed:.2f} seconds ({total_images / max(elapsed, 0.1):.2f} img/sec).")
    print(f"Segmentation maps saved to: {os.path.abspath(args.output_dir)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU Parallel SAM / FastSAM Segment Generator")
    parser.add_argument("--input_dir", type=str, default=None, help="Path to input dataset image directory")
    parser.add_argument("--flist", type=str, default=None, help="Path to flist file containing list of image paths")
    parser.add_argument("--output_dir", type=str, default="./sam_segmentations", help="Directory to save segment maps")
    
    parser.add_argument("--backend", type=str, choices=["ultralytics", "meta"], default="ultralytics",
                        help="SAM backend library to use (ultralytics or meta)")
    parser.add_argument("--model_path", type=str, default="fastsam-s.pt",
                        help="Path or weight name for SAM model (e.g. fastsam-s.pt, sam2.1_t.pt, sam_b.pt, mobile_sam.pt)")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"],
                        help="Meta SAM model type (only if backend=meta)")
    
    parser.add_argument("--gpus", nargs="+", default=["auto"], help="List of GPU device IDs to use, e.g., --gpus 0 1")
    parser.add_argument("--workers_per_gpu", type=int, default=4, help="Number of parallel worker processes per GPU")
    parser.add_argument("--num_save_workers", type=int, default=8, help="CPU thread pool size for background PNG saving per worker")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 Automatic Mixed Precision")
    parser.add_argument("--no_fp16", action="store_false", dest="fp16", help="Disable FP16 AMP")
    
    parser.add_argument("--width", type=int, default=None, help="Target width to resize segmentation maps (optional)")
    parser.add_argument("--height", type=int, default=None, help="Target height to resize segmentation maps (optional)")
    parser.add_argument("--save_color_map", action="store_true", help="Save visual RGB colorized segment images")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()
