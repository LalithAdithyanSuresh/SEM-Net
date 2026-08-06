#!/usr/bin/env python3
import os
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
    
    # Load Model
    model = InpaintingModel(config)
    model.gen_weights_path = gen_checkpoint
    model.load()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    model = model.to(config.DEVICE)
    model.eval()

    # Prepare Dataset
    test_dataset = Dataset(config, args.image_dir, args.mask_dir, augment=False, training=False)
    if len(test_dataset) == 0:
        raise ValueError(f"No validation images found in {args.image_dir}.")
        
    print(f"Validation dataset has {len(test_dataset)} unique images.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Index custom masks
    indexed_masks = index_custom_masks(args.mask_dir)
    categories = ['SMALL', 'MEDIUM', 'LARGE']
    
    # Check if there are masks
    for cat in categories:
        if not indexed_masks[cat]:
            print(f"WARNING: No masks found for category {cat}. Results will be filled with zeros.")

    # Initialize loss metric
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)
    loss_fn_vgg.eval()

    # Create directories
    create_dir(args.output)
    for cat in categories:
        create_dir(os.path.join(args.output, 'images', cat))

    executor = ThreadPoolExecutor(max_workers=4)

    # Main evaluation loop per category
    for cat in categories:
        print(f"\n==========================================")
        print(f"Evaluating category: {cat} (Target: {args.num_images} images)")
        print(f"==========================================")
        
        stats = {'name': [], 'psnr': [], 'ssim': [], 'l1': [], 'lpips': []}
        
        # Enumerate and generate up to num_images
        pbar = tqdm(total=args.num_images, desc=f"Generating {cat}")
        
        count = 0
        dataset_len = len(test_dataset)
        
        while count < args.num_images:
            for index, items in enumerate(test_loader):
                if count >= args.num_images:
                    break
                    
                images, _, _ = items
                images = images.to(config.DEVICE)
                h, w = images.shape[2], images.shape[3]
                
                # Fetch custom mask for this index
                masks = get_custom_mask(indexed_masks, cat, count, h, w, images.size(0)).to(config.DEVICE)
                
                with torch.no_grad():
                    outputs_img = model(images, masks)
                    
                outputs_merged = (outputs_img * masks) + (images * (1 - masks))
                
                # Calculate metrics for each item in the batch
                for b in range(images.size(0)):
                    if count >= args.num_images:
                        break
                        
                    psnr, ssim = calc_psnr_ssim(images[b:b+1], outputs_merged[b:b+1])
                    l1_val = F.l1_loss(outputs_merged[b:b+1], images[b:b+1], reduction='mean').item()
                    lpips_val = loss_fn_vgg(normalize_lpips(outputs_merged[b:b+1]), normalize_lpips(images[b:b+1])).item()
                    
                    # Determine filename (handling loop wraparound)
                    orig_idx = (count) % dataset_len
                    orig_name = test_dataset.load_name(orig_idx)
                    base, ext = os.path.splitext(orig_name)
                    file_name = f"{base}_{count}{ext}"
                    
                    stats['name'].append(file_name)
                    stats['psnr'].append(psnr)
                    stats['ssim'].append(ssim)
                    stats['l1'].append(l1_val)
                    stats['lpips'].append(lpips_val)
                    
                    # Save generated image asynchronously
                    merged_img_pil = Image.fromarray(postprocess(outputs_merged[b:b+1])[0].cpu().numpy().astype(np.uint8))
                    save_path = os.path.join(args.output, 'images', cat, file_name)
                    executor.submit(save_task, save_path, merged_img_pil)
                    
                    count += 1
                    pbar.update(1)
                    
        pbar.close()
        
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

    executor.shutdown(wait=True)
    print(f"\nAll done! Results saved to {args.output}")

if __name__ == '__main__':
    main()
