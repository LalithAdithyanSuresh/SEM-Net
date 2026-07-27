#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

def get_out_path(img_path, output_dir):
    filename = os.path.basename(img_path)
    base_name, _ = os.path.splitext(filename)
    return os.path.join(output_dir, f"{base_name}.png")

def _gpu_worker(gpu_id, image_paths, output_dir, model_name, batch_size, status_queue):
    """
    Worker process running on target GPU/CPU.
    Loads FastSAM model once on assigned device and processes image paths in batches.
    """
    try:
        device = f"cuda:{gpu_id}" if (isinstance(gpu_id, int) and torch.cuda.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")
        from ultralytics import FastSAM
        model = FastSAM(model_name)
        
        # Filter out images that are already processed
        unprocessed = []
        for img_path in image_paths:
            out_path = get_out_path(img_path, output_dir)
            if os.path.exists(out_path):
                status_queue.put(1)
            else:
                unprocessed.append(img_path)

        for i in range(0, len(unprocessed), batch_size):
            batch_paths = unprocessed[i:i + batch_size]
            try:
                results = model(batch_paths, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, verbose=False)
                for img_path, res in zip(batch_paths, results):
                    out_path = get_out_path(img_path, output_dir)
                    if res is not None and res.masks is not None and len(res.masks.data) > 0:
                        masks = res.masks.data.cpu().numpy()
                        combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
                    else:
                        img = cv2.imread(img_path)
                        h, w = img.shape[:2] if img is not None else (256, 256)
                        combined_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.imwrite(out_path, combined_mask)
                    status_queue.put(1)
            except Exception as batch_err:
                # Fallback to 1-by-1 processing if batch execution fails
                for img_path in batch_paths:
                    out_path = get_out_path(img_path, output_dir)
                    try:
                        res_single = model(img_path, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, verbose=False)
                        if res_single and len(res_single) > 0 and res_single[0].masks is not None and len(res_single[0].masks.data) > 0:
                            masks = res_single[0].masks.data.cpu().numpy()
                            combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
                        else:
                            img = cv2.imread(img_path)
                            h, w = img.shape[:2] if img is not None else (256, 256)
                            combined_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.imwrite(out_path, combined_mask)
                    except Exception as single_err:
                        print(f"[{device}] Error processing {img_path}: {single_err}", file=sys.stderr)
                    status_queue.put(1)
    except Exception as e:
        print(f"Worker exception on gpu {gpu_id}: {e}", file=sys.stderr)

def generate_masks_for_dir(input_dir, output_dir, gpus=None, batch_size=16, model_name="FastSAM-s.pt"):
    if not os.path.exists(input_dir):
        print(f"[FastSAM] Input directory {input_dir} does not exist. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    image_paths = sorted(list(set(image_paths)))
    if not image_paths:
        print(f"[FastSAM] No images found in {input_dir}.")
        return

    if gpus is None:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_gpus))
        else:
            gpu_list = ["cpu"]
    else:
        gpu_list = gpus

    num_workers = len(gpu_list)
    print(f"[FastSAM] Processing {len(image_paths)} images from {input_dir} -> {output_dir}")
    print(f"[FastSAM] Using {num_workers} worker process(es) across GPU(s): {gpu_list} (Batch size: {batch_size})")

    # Chunk image paths across workers
    chunks = [[] for _ in range(num_workers)]
    for idx, path in enumerate(image_paths):
        chunks[idx % num_workers].append(path)

    ctx = mp.get_context("spawn")
    status_queue = ctx.Queue()
    processes = []

    for gpu_id, chunk in zip(gpu_list, chunks):
        if not chunk:
            continue
        p = ctx.Process(
            target=_gpu_worker,
            args=(gpu_id, chunk, output_dir, model_name, batch_size, status_queue)
        )
        p.start()
        processes.append(p)

    with tqdm(total=len(image_paths), desc=f"Segmenting {os.path.basename(output_dir)}") as pbar:
        completed = 0
        while completed < len(image_paths):
            try:
                status_queue.get(timeout=1.0)
                completed += 1
                pbar.update(1)
            except Exception:
                if not any(p.is_alive() for p in processes) and status_queue.empty():
                    break

    for p in processes:
        p.join()

    print(f"[FastSAM] Completed segment mask generation for {output_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Generate segment masks using FastSAM with multi-GPU parallel processing.")
    parser.add_argument("--dataset-dir", default="dataset", help="Root dataset directory containing train/test folders")
    parser.add_argument("--train-dir", default=None, help="Train image folder path")
    parser.add_argument("--test-dir", default=None, help="Test image folder path")
    parser.add_argument("--model", default="FastSAM-s.pt", help="FastSAM model weights file")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU inference call (default: 16)")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU indices to use, e.g., '0,1' (default: auto-detect all GPUs)")
    args = parser.parse_args()

    gpus = None
    if args.gpus is not None:
        gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip().isdigit()]

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
    print(f"  Batch size: {args.batch_size}")

    if os.path.exists(train_input):
        generate_masks_for_dir(train_input, train_output, gpus=gpus, batch_size=args.batch_size, model_name=args.model)
    
    if os.path.exists(test_input):
        generate_masks_for_dir(test_input, test_output, gpus=gpus, batch_size=args.batch_size, model_name=args.model)

if __name__ == "__main__":
    main()
