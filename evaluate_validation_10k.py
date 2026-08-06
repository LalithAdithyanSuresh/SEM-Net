#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import Dataset
from src.models import InpaintingModel
from src.utils import create_dir
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
import torchvision
from PIL import Image
import csv
import time
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- CUSTOM MASK INDEXING ---
import torch.multiprocessing as mp
import json

def index_custom_masks(mask_dir):
    print(f"Indexing masks in {mask_dir}...")
    categories = {'SMALL': [], 'MEDIUM': [], 'LARGE': []}
    if not os.path.exists(mask_dir):
        print(f"WARNING: Mask directory {mask_dir} does not exist.")
        return categories
        
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    mask_files.sort()
    
    for f in tqdm(mask_files, desc="Indexing masks"):
        mask_path = os.path.join(mask_dir, f)
        try:
            mask_img = Image.open(mask_path).convert('L')
            mask_np = np.array(mask_img)
            ratio = np.mean(mask_np) / 255.0
            
            if 0.01 < ratio <= 0.20:
                categories['SMALL'].append(mask_path)
            elif 0.20 < ratio <= 0.40:
                categories['MEDIUM'].append(mask_path)
            elif 0.40 < ratio <= 0.60:
                categories['LARGE'].append(mask_path)
        except Exception as e:
            print(f"Skip {f}: {e}")
            
    print(f"Index complete: SMALL({len(categories['SMALL'])}), MEDIUM({len(categories['MEDIUM'])}), LARGE({len(categories['LARGE'])})")
    return categories

# --- MASK LOADING ---
def get_custom_mask(indexed_masks, cat, index, h, w, batch_size):
    if not indexed_masks[cat]:
        return torch.zeros((batch_size, 1, h, w))
        
    mask_tensors = []
    for i in range(batch_size):
        mask_idx = (index * batch_size + i) % len(indexed_masks[cat])
        mask_path = indexed_masks[cat][mask_idx]
        mask_img = Image.open(mask_path).convert('L').resize((w, h), Image.NEAREST)
        mask_tensor = torchvision.transforms.functional.to_tensor(mask_img).float()
        mask_tensors.append(mask_tensor)
        
    return torch.stack(mask_tensors)

# --- POSTPROCESS ---
def postprocess(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()

# --- METRICS ---
def calc_psnr_ssim(gt, pre):
    pre = pre.clamp(0, 1) * 255.0
    pre = pre.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]

    gt = gt.clamp(0, 1) * 255.0
    gt = gt.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]

    psnr = min(100, compare_psnr(gt, pre, data_range=255))
    ssim = compare_ssim(gt, pre, channel_axis=-1, data_range=255)
    return psnr, ssim

# --- LPIPS NORMALIZATION ---
def normalize_lpips(x):
    return (x * 2) - 1

# --- ASYNC SAVE TASK ---
def save_task(path, img):
    try:
        img.save(path)
    except Exception as e:
        print(f"Error saving image {path}: {e}")

# --- FIND LATEST CHECKPOINT ---
def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('_gen.pth')]
    if not checkpoint_files:
        return None
        
    # Attempt to sort by iteration/epoch numbers extracted from filenames
    best_file = None
    max_val = -1
    
    for f in checkpoint_files:
        # Match iteration first: e.g. InpaintingModel_top_psnr_34.99_iter_392000_gen.pth
        iter_match = re.search(r'iter_(\d+)', f)
        if iter_match:
            val = int(iter_match.group(1))
            if val > max_val:
                max_val = val
                best_file = f
                continue
                
        # Match epoch next: e.g. InpaintingModel_best_epoch_57_gen.pth
        epoch_match = re.search(r'epoch_(\d+)', f)
        if epoch_match and max_val == -1: # Iteration takes precedence, but if not found, use epoch
            val = int(epoch_match.group(1))
            if val > max_val:
                max_val = val
                best_file = f

    # Fallback to sorting by file modification time if no numeric patterns matched
    if best_file is None:
        checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        best_file = checkpoint_files[-1]
        
    return os.path.join(checkpoint_dir, best_file)

def main():
    parser = argparse.ArgumentParser(description="Evaluate SEM-Net model and generate validation images/reports.")
    parser.add_argument('--path', type=str, default='./segmentCamino_Finished_33.5PSNR', help='Directory containing the checkpoints and config.yml')
    parser.add_argument('--checkpoint', type=str, default=None, help='Direct path to a specific generator checkpoint (.pth)')
    parser.add_argument('--mask-dir', type=str, default='datasets/testing_mask_dataset', help='Directory with the custom masks')
    parser.add_argument('--image-dir', type=str, default='datasets/places365/test_256', help='Directory with validation/testing images')
    parser.add_argument('--output', type=str, default='./evaluation_results_10k', help='Output directory')
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    parser.add_argument('--batch-size', type=int, default=4 * num_gpus, help='Batch size for inference')
    parser.add_argument('--num-images', type=int, default=10000, help='Number of validation images to generate per mask category')
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.join(args.path, 'config.yml')
    if not os.path.exists(config_path):
        if os.path.exists('./config.yml'):
            config_path = './config.yml'
        else:
            raise FileNotFoundError("Could not locate config.yml. Provide a valid --path argument containing config.yml.")
            
    config = Config(config_path)
    config.PATH = args.path
    config.MODE = 2
    config.MODEL = 2
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.WORLD_SIZE = 1

    # Find the generator checkpoint
    if args.checkpoint is not None:
        gen_checkpoint = args.checkpoint
    else:
        gen_checkpoint = find_latest_checkpoint(args.path)
        
    if gen_checkpoint is None or not os.path.exists(gen_checkpoint):
        raise FileNotFoundError(f"Could not find any valid generator checkpoint in {args.path}. Please verify --path or --checkpoint.")
        
    print(f"Using generator checkpoint: {gen_checkpoint}")
    print(f"CUDA devices available: {num_gpus}")
    
    # Prepare Dataset
    test_dataset = Dataset(config, args.image_dir, args.mask_dir, augment=False, training=False)
    if len(test_dataset) == 0:
        raise ValueError(f"No validation images found in {args.image_dir}.")
        
    print(f"Validation dataset has {len(test_dataset)} unique images.")

    # Index custom masks
    indexed_masks = index_custom_masks(args.mask_dir)
    categories = ['SMALL', 'MEDIUM', 'LARGE']
    
    # Check if there are masks
    for cat in categories:
        if not indexed_masks[cat]:
            print(f"WARNING: No masks found for category {cat}. Results will be filled with zeros.")

    # Create directories
    create_dir(args.output)
    for cat in categories:
        create_dir(os.path.join(args.output, 'images', cat))

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    stats_dict = manager.dict()
    
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker, args=(gpu_id, num_gpus, args, config, gen_checkpoint, indexed_masks, categories, stats_dict))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    # Main evaluation loop per category
    for cat in categories:
        stats = {'name': [], 'psnr': [], 'ssim': [], 'l1': [], 'lpips': []}
        for gpu_id in range(num_gpus):
            gpu_stats = stats_dict.get(f"{cat}_{gpu_id}", {'name': [], 'psnr': [], 'ssim': [], 'l1': [], 'lpips': []})
            for k in stats:
                stats[k].extend(gpu_stats[k])
                
        def get_index_from_name(name):
            match = re.search(r'_(\d+)\.[^.]+$', name)
            return int(match.group(1)) if match else 0
            
        sorted_indices = np.argsort([get_index_from_name(name) for name in stats['name']])
        for k in stats:
            stats[k] = [stats[k][idx] for idx in sorted_indices]
        
        # Write CSV report for this category
        csv_path = os.path.join(args.output, f'metrics_{cat}.csv')
        print(f"Saving CSV report to {csv_path}...")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'PSNR', 'SSIM', 'L1', 'LPIPS'])
            for i in range(len(stats['name'])):
                writer.writerow([
                    stats['name'][i],
                    f"{stats['psnr'][i]:.4f}",
                    f"{stats['ssim'][i]:.4f}",
                    f"{stats['l1'][i]:.6f}",
                    f"{stats['lpips'][i]:.6f}"
                ])
            
            writer.writerow([])
            writer.writerow([
                'AVERAGE',
                f"{np.mean(stats['psnr']):.4f}",
                f"{np.mean(stats['ssim']):.4f}",
                f"{np.mean(stats['l1']):.6f}",
                f"{np.mean(stats['lpips']):.6f}"
            ])
            
        print(f"{cat} average stats:")
        print(f"  PSNR:  {np.mean(stats['psnr']):.4f}")
        print(f"  SSIM:  {np.mean(stats['ssim']):.4f}")
        print(f"  L1:    {np.mean(stats['l1']):.6f}")
        print(f"  LPIPS: {np.mean(stats['lpips']):.6f}")

    print(f"\nAll done! Results saved to {args.output}")

if __name__ == '__main__':
    main()
