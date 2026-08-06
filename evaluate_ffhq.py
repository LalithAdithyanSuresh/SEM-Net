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
 
    # Override image flists for evaluate script
    config.TEST_INPAINT_IMAGE_FLIST = os.path.join(args.dataset_root, 'test')
    config.TEST_MASK_FLIST = os.path.join(args.dataset_root, 'masks')
 
    # LPIPS
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)
    loss_fn_vgg.eval()
 
    # Find the generator checkpoint
    if args.checkpoint is not None:
        gen_checkpoint = args.checkpoint
    else:
        gen_checkpoint = find_latest_checkpoint(args.path)
        
    if gen_checkpoint is None or not os.path.exists(gen_checkpoint):
        raise FileNotFoundError(f"Could not find any valid generator checkpoint in {args.path}.")
        
    print(f"Using generator checkpoint: {gen_checkpoint}")
    print(f"CUDA devices available: {torch.cuda.device_count()}")
 
    # Model
    model = InpaintingModel(config)
    model.gen_weights_path = gen_checkpoint
    model.load()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    model = model.to(config.DEVICE)
    model.eval()
 
    # Load custom Dataset
    test_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST, config.TEST_MASK_FLIST,
                           augment=False, training=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
 
    # Index Custom Masks
    mask_dir = os.path.join(args.dataset_root, 'masks')
    indexed_masks = index_custom_masks(mask_dir)
 
    categories = ['SMALL', 'MEDIUM', 'LARGE']
    create_dir(args.output)
 
    executor = ThreadPoolExecutor(max_workers=4)
 
    # Process each category
    for cat in categories:
        print(f"\n==========================================")
        print(f"Evaluating category: {cat} (Target: {args.num_images} images)")
        print(f"==========================================")
        cat_output_dir = os.path.join(args.output, cat)
        create_dir(cat_output_dir)
        
        csv_path = os.path.join(args.output, f'metrics_{cat}.csv')
        stats = {'name': [], 'mask_id': [], 'psnr': [], 'ssim': [], 'l1': [], 'lpips': []}
 
        pbar = tqdm(total=args.num_images, desc=f"Eval {cat}")
        count = 0
        dataset_len = len(test_dataset)
 
        while count < args.num_images:
            for index, items in enumerate(test_loader):
                if count >= args.num_images:
                    break
                
                # Retrieve items
                images, _, _ = items
                images = images.to(config.DEVICE)
                h, w = images.shape[2], images.shape[3]
                curr_batch_size = images.size(0)
                
                # Load custom mask in rotation
                masks, mask_ids = get_custom_masks_and_ids(indexed_masks, cat, count, h, w, curr_batch_size)
                masks = masks.to(config.DEVICE)
 
                # Manually load the segment maps from the overridden segment directory
                seg_tensors = []
                file_names = []
                image_ids = []
                for i in range(curr_batch_size):
                    orig_idx = (count + i) % dataset_len
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
                            seg_tensor = torchvision.transforms.functional.to_tensor(seg_img).to(config.DEVICE)
                        except Exception:
                            seg_tensor = torch.zeros((1, h, w), device=config.DEVICE)
                    else:
                        seg_tensor = torch.zeros((1, h, w), device=config.DEVICE)
                    seg_tensors.append(seg_tensor)
                
                seg_maps = torch.stack(seg_tensors)
 
                with torch.no_grad():
                    outputs_img = model(images, masks, seg_maps=seg_maps)
 
                outputs_merged = (outputs_img * masks) + (images * (1 - masks))
 
                for b in range(curr_batch_size):
                    if count >= args.num_images:
                        break
                    
                    # Metrics
                    psnr, ssim = calc_psnr_ssim(images[b:b+1], outputs_merged[b:b+1])
                    l1_val = F.l1_loss(outputs_merged[b:b+1], images[b:b+1], reduction='mean').item()
                    lpips_val = loss_fn_vgg(normalize_lpips(outputs_merged[b:b+1]), normalize_lpips(images[b:b+1])).item()
 
                    # Save metrics (handling name duplicate formatting for wrapping)
                    base, ext = os.path.splitext(file_names[b])
                    saved_filename = f"{base}_{count}{ext}"
 
                    stats['name'].append(saved_filename)
                    stats['mask_id'].append(mask_ids[b])
                    stats['psnr'].append(psnr)
                    stats['ssim'].append(ssim)
                    stats['l1'].append(l1_val)
                    stats['lpips'].append(lpips_val)
 
                    # Save generated image with exact naming convention: imageID_maskID_PSNR.png
                    pred_merged_pil = Image.fromarray(postprocess(outputs_merged[b:b+1])[0].cpu().numpy().astype(np.uint8))
                    save_name = f"{image_ids[b]}_{mask_ids[b]}_{psnr:.2f}.png"
                    executor.submit(save_task, os.path.join(cat_output_dir, save_name), pred_merged_pil)
 
                    count += 1
                    pbar.update(1)

        pbar.close()

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

    executor.shutdown(wait=True)
    print(f"All done! Results saved in {args.output}")

if __name__ == '__main__':
    main()
