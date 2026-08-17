#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
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
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import torch.multiprocessing as mp
import json


def postprocess(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()

def calc_psnr_ssim(gt, pre):
    # gt and pre are 3-channel RGB tensors [1, 3, H, W]
    pre = pre.clamp(0, 1) * 255.0
    pre = pre.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
    gt = gt.clamp(0, 1) * 255.0
    gt = gt.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
    psnr = min(100, compare_psnr(gt, pre, data_range=255))
    ssim = compare_ssim(gt, pre, channel_axis=-1, data_range=255)
    return psnr, ssim

def normalize_lpips(x):
    return (x * 2) - 1

def index_custom_masks(mask_dir):
    print(f"Indexing masks in {mask_dir}...")
    categories = {'SMALL': [], 'MEDIUM': [], 'LARGE': []}
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    mask_files.sort()
    
    for f in tqdm(mask_files, desc="Indexing masks"):
        mask_path = os.path.join(mask_dir, f)
        try:
            mask_img = Image.open(mask_path).convert('L')
            mask_np = np.array(mask_img)
            ratio = np.mean(mask_np) / 255.0
            
            if 0.01 < ratio <= 0.20:
                categories['SMALL'].append((f, mask_path))
            elif 0.20 < ratio <= 0.40:
                categories['MEDIUM'].append((f, mask_path))
            elif 0.40 < ratio <= 0.60:
                categories['LARGE'].append((f, mask_path))
        except Exception as e:
            print(f"Skip {f}: {e}")
            
    print(f"Index complete: SMALL({len(categories['SMALL'])}), MEDIUM({len(categories['MEDIUM'])}), LARGE({len(categories['LARGE'])})")
    return categories

def get_custom_masks_and_ids(indexed_masks, cat, count, h, w, batch_size):
    mask_list = indexed_masks[cat]
    if not mask_list:
        return torch.zeros((batch_size, 1, h, w)), ["no_mask"] * batch_size
    
    mask_tensors = []
    mask_ids = []
    for i in range(batch_size):
        mask_name, mask_path = mask_list[(count + i) % len(mask_list)]
        mask_img = Image.open(mask_path).convert('L').resize((w, h), Image.NEAREST)
        mask_tensor = torchvision.transforms.functional.to_tensor(mask_img).float()
        mask_tensors.append(mask_tensor)
        mask_ids.append(os.path.splitext(mask_name)[0])
    return torch.stack(mask_tensors), mask_ids

# --- FIND LATEST CHECKPOINT ---
def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('_gen.pth')]
    if not checkpoint_files:
        return None
        
    best_file = None
    max_val = -1
    
    for f in checkpoint_files:
        iter_match = re.search(r'iter_(\d+)', f)
        if iter_match:
            val = int(iter_match.group(1))
            if val > max_val:
                max_val = val
                best_file = f
                continue
                
        epoch_match = re.search(r'epoch_(\d+)', f)
        if epoch_match and max_val == -1:
            val = int(epoch_match.group(1))
            if val > max_val:
                max_val = val
                best_file = f
 
    if best_file is None:
        checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        best_file = checkpoint_files[-1]
        
    return os.path.join(checkpoint_dir, best_file)
 
# --- ASYNC SAVE TASK ---
def save_task(path, img):
    try:
        img.save(path)
    except Exception as e:
        print(f"Error saving image {path}: {e}")

def worker(gpu_id, num_gpus, args, config_path, gen_checkpoint, indexed_masks, categories):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    config = Config(config_path)
    config.PATH = args.path
    config.MODE = 2
    config.MODEL = 2
    config.DEVICE = device
    config.WORLD_SIZE = 1
    config.TEST_INPAINT_IMAGE_FLIST = os.path.join(args.dataset_root, 'test')
    config.TEST_MASK_FLIST = os.path.join(args.dataset_root, 'masks')
    
    # Load Model and Loss
    model = InpaintingModel(config).to(device)
    model.gen_weights_path = gen_checkpoint
    model.load()
    model.eval()
    
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    loss_fn_vgg.eval()
    
    test_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST, config.TEST_MASK_FLIST,
                           augment=False, training=False)
    dataset_len = len(test_dataset)
    executor = ThreadPoolExecutor(max_workers=4)
    
    batch_size_per_gpu = max(1, args.batch_size // num_gpus)
    
    all_gpu_stats = {}
    for cat in categories:
        cat_output_dir = os.path.join(args.output, cat)
        
        global_counts = list(range(gpu_id * batch_size_per_gpu, args.num_images, num_gpus * batch_size_per_gpu))
        local_stats = {'name': [], 'mask_id': [], 'psnr': [], 'ssim': [], 'l1': [], 'lpips': []}
        
        pbar = tqdm(total=args.num_images, desc=f"Eval {cat}") if gpu_id == 0 else None
        
        for start_count in global_counts:
            batch_global_indices = []
            for b_idx in range(batch_size_per_gpu):
                if start_count + b_idx < args.num_images:
                    batch_global_indices.append(start_count + b_idx)
            
            if not batch_global_indices:
                continue
                
            image_list = []
            for g_idx in batch_global_indices:
                orig_idx = g_idx % dataset_len
                img, _, _ = test_dataset[orig_idx]
                image_list.append(img)
                
            images = torch.stack(image_list).to(device)
            h, w = images.shape[2], images.shape[3]
            curr_batch_size = images.size(0)
            
            masks, mask_ids = get_custom_masks_and_ids(indexed_masks, cat, batch_global_indices[0], h, w, curr_batch_size)
            masks = masks.to(device)
            
            seg_tensors = []
            file_names = []
            image_ids = []
            for i, g_idx in enumerate(batch_global_indices):
                orig_idx = g_idx % dataset_len
                file_name = test_dataset.load_name(orig_idx)
                file_names.append(file_name)
                image_id = os.path.splitext(file_name)[0]
                image_ids.append(image_id)
                
                seg_path = os.path.join(args.seg_dir, f"{image_id}.png")
                if os.path.exists(seg_path):
                    try:
                        seg_img = Image.open(seg_path)
                        if seg_img.mode != 'L':
                            seg_img = seg_img.convert('L')
                        seg_img = seg_img.resize((w, h), Image.NEAREST)
                        seg_tensor = torchvision.transforms.functional.to_tensor(seg_img).to(device)
                    except Exception:
                        seg_tensor = torch.zeros((1, h, w), device=device)
                else:
                    seg_tensor = torch.zeros((1, h, w), device=device)
                seg_tensors.append(seg_tensor)
                
            seg_maps = torch.stack(seg_tensors)
            
            with torch.no_grad():
                outputs_img = model(images, masks, seg_maps=seg_maps)
                
            outputs_merged = (outputs_img * masks) + (images * (1 - masks))
            
            for b in range(curr_batch_size):
                psnr, ssim = calc_psnr_ssim(images[b:b+1], outputs_merged[b:b+1])
                l1_val = F.l1_loss(outputs_merged[b:b+1], images[b:b+1], reduction='mean').item()
                lpips_val = loss_fn_vgg(normalize_lpips(outputs_merged[b:b+1]), normalize_lpips(images[b:b+1])).item()
                
                base, ext = os.path.splitext(file_names[b])
                saved_filename = f"{base}_{batch_global_indices[b]}{ext}"
                
                local_stats['name'].append(saved_filename)
                local_stats['mask_id'].append(mask_ids[b])
                local_stats['psnr'].append(psnr)
                local_stats['ssim'].append(ssim)
                local_stats['l1'].append(l1_val)
                local_stats['lpips'].append(lpips_val)
                
                pred_merged_pil = Image.fromarray(postprocess(outputs_merged[b:b+1])[0].cpu().numpy().astype(np.uint8))
                save_name = f"{image_ids[b]}_{mask_ids[b]}_{psnr:.2f}.png"
                executor.submit(save_task, os.path.join(cat_output_dir, save_name), pred_merged_pil)
                
                if pbar:
                    pbar.update(num_gpus)
                    
        if pbar:
            pbar.close()
            
        all_gpu_stats[cat] = local_stats
        
    executor.shutdown(wait=True)
    
    import json
    temp_path = os.path.join(args.output, f".tmp_stats_gpu_{gpu_id}.json")
    with open(temp_path, 'w') as f:
        json.dump(all_gpu_stats, f)
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./segmentCamino_Finished_33.5PSNR')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to specific generator checkpoint')
    parser.add_argument('--output', type=str, default='./evaluation_results_ffhq')
    parser.add_argument('--dataset_root', type=str, default='./dataset')
    parser.add_argument('--seg_dir', type=str, default='./dataset/test_seg')
    parser.add_argument('--num-images', type=int, default=10000, help='Number of validation images per category')
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    parser.add_argument('--batch-size', type=int, default=4 * num_gpus, help='Batch size for evaluation')
    args = parser.parse_args()
 
    config_path = os.path.join(args.path, 'config.yml')
    if not os.path.exists(config_path):
        if os.path.exists('./config.yml'):
            config_path = './config.yml'
        else:
            raise FileNotFoundError("Could not locate config.yml.")
            
    config = Config(config_path)
    config.PATH = args.path
    config.MODE = 2
    config.MODEL = 2
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.WORLD_SIZE = 1
 
    config.TEST_INPAINT_IMAGE_FLIST = os.path.join(args.dataset_root, 'test')
    config.TEST_MASK_FLIST = os.path.join(args.dataset_root, 'masks')
 
    if args.checkpoint is not None:
        gen_checkpoint = args.checkpoint
    else:
        gen_checkpoint = find_latest_checkpoint(args.path)
        
    if gen_checkpoint is None or not os.path.exists(gen_checkpoint):
        raise FileNotFoundError(f"Could not find any valid generator checkpoint in {args.path}.")
        
    print(f"Using generator checkpoint: {gen_checkpoint}")
    print(f"CUDA devices available: {num_gpus}")
 
    # Index Custom Masks
    mask_dir = os.path.join(args.dataset_root, 'masks')
    indexed_masks = index_custom_masks(mask_dir)
 
    categories = ['SMALL', 'MEDIUM', 'LARGE']
    create_dir(args.output)
    for cat in categories:
        create_dir(os.path.join(args.output, cat))
 
    mp.spawn(worker, args=(num_gpus, args, config_path, gen_checkpoint, indexed_masks, categories), nprocs=num_gpus)
 
    # Process each category to save metrics
    for cat in categories:
        csv_path = os.path.join(args.output, f'metrics_{cat}.csv')
        stats = {'name': [], 'mask_id': [], 'psnr': [], 'ssim': [], 'l1': [], 'lpips': []}
        
        for gpu_id in range(num_gpus):
            temp_path = os.path.join(args.output, f".tmp_stats_gpu_{gpu_id}.json")
            if os.path.exists(temp_path):
                with open(temp_path, 'r') as f:
                    gpu_stats = json.load(f)
                if cat in gpu_stats:
                    for k in stats:
                        stats[k].extend(gpu_stats[cat][k])
                
        def get_index_from_name(name):
            match = re.search(r'_(\d+)\.[^.]+$', name)
            return int(match.group(1)) if match else 0
            
        sorted_indices = np.argsort([get_index_from_name(name) for name in stats['name']])
        for k in stats:
            stats[k] = [stats[k][idx] for idx in sorted_indices]
 
        # Write CSV for this category
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'MaskID', 'PSNR', 'SSIM', 'L1', 'LPIPS'])
            for i in range(len(stats['name'])):
                writer.writerow([
                    stats['name'][i],
                    stats['mask_id'][i],
                    f"{stats['psnr'][i]:.4f}",
                    f"{stats['ssim'][i]:.4f}",
                    f"{stats['l1'][i]:.6f}",
                    f"{stats['lpips'][i]:.6f}"
                ])
            
            writer.writerow([])
            writer.writerow([
                'AVERAGE',
                '',
                f"{np.mean(stats['psnr']):.4f}",
                f"{np.mean(stats['ssim']):.4f}",
                f"{np.mean(stats['l1']):.6f}",
                f"{np.mean(stats['lpips']):.6f}"
            ])
 
        print(f"Category {cat} finalized. Average PSNR: {np.mean(stats['psnr']):.2f}")
 
    # Clean up temporary JSON files
    for gpu_id in range(num_gpus):
        temp_path = os.path.join(args.output, f".tmp_stats_gpu_{gpu_id}.json")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Error removing {temp_path}: {e}")
 
    print(f"All done! Results saved in {args.output}")

if __name__ == '__main__':
    main()
