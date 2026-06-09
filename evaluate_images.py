import os
try:
    import getpass
    username = getpass.getuser()
except Exception:
    username = "semnet_user"
os.environ["TORCH_HOME"] = f"/tmp/{username}/torch_cache"
os.environ["MPLCONFIGDIR"] = f"/tmp/{username}/matplotlib_cache"

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
from cleanfid import fid
from scipy import linalg
import csv
import time
import requests
import shutil
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# --- NTFY NOTIFICATIONS ---
def send_notification(message):
    try:
        topic = "validation-images-places"
        requests.post(f"https://ntfy.sh/{topic}", data=message.encode(encoding='utf-8'), timeout=5)
        print(f"Sent push notification: {message}")
    except Exception as e:
        print(f"Failed to send push notification: {e}")

# --- PATCH CLEANFID FRECHET DISTANCE ---
def robust_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
    if np.iscomplexobj(covmean):
        m = np.max(np.abs(covmean.imag))
        if m > 1e-3:
            print(f"Warning: Imaginary component {m} found in FID calculation. Taking real part.")
        covmean = covmean.real
        
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

fid.frechet_distance = robust_frechet_distance

# --- CUSTOM MASK INDEXING ---
def index_custom_masks(mask_dir):
    print(f"Indexing masks in {mask_dir}...")
    categories = {'SMALL': [], 'MEDIUM': [], 'LARGE': []}
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    mask_files.sort() # Ensure deterministic sequential order
    
    for f in tqdm(mask_files):
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

# --- MASK LOADING (EVEN ONLY STRIDE=2) ---
def get_custom_mask(indexed_masks, cat, index, h, w, batch_size, stride=2):
    if not indexed_masks[cat]:
        return torch.zeros((batch_size, 1, h, w))
        
    mask_tensors = []
    for i in range(batch_size):
        # Selection with stride=2 selects even-indexed masks
        mask_idx = ((index * batch_size + i) * stride) % len(indexed_masks[cat])
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
def save_task(path, img, compress_level=1):
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.png':
            img.save(path, compress_level=compress_level)
        elif ext in ['.jpg', '.jpeg']:
            img.save(path, quality=95)
        else:
            img.save(path)
    except Exception as e:
        print(f"Error saving image {path}: {e}")

# --- CHUNKED UPLOAD ---
def upload_file_chunked(file_path, server_url, session_id, chunk_size=10 * 1024 * 1024):
    if not os.path.exists(file_path):
        print(f"[UPLOAD] File {file_path} not found.")
        return False
        
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    total_chunks = (file_size + chunk_size - 1) // chunk_size
    
    print(f"[UPLOAD] Uploading {filename} ({file_size / (1024*1024):.2f} MB) in {total_chunks} chunks...")
    print(f"[UPLOAD] Destination URL: {server_url}/api/upload_chunk")
    
    try:
        with open(file_path, 'rb') as f:
            for i in range(total_chunks):
                chunk_data = f.read(chunk_size)
                files = {'file': (f"{filename}.part{i}", chunk_data, 'application/octet-stream')}
                data = {
                    'session': session_id,
                    'filename': filename,
                    'chunk_index': i,
                    'total_chunks': total_chunks
                }
                
                success = False
                for retry in range(3):
                    try:
                        res = requests.post(f"{server_url}/api/upload_chunk", files=files, data=data, timeout=60)
                        print(f"  -> Chunk {i+1}/{total_chunks}: Server returned status {res.status_code}")
                        if res.status_code == 200:
                            success = True
                            break
                        else:
                            print(f"  -> Response content: {res.text[:300]}")
                    except Exception as e:
                        print(f"  -> Chunk {i+1}/{total_chunks} retry {retry+1} error: {e}")
                    time.sleep(1)
                    
                if not success:
                    print(f"[UPLOAD] Failed to upload chunk {i+1}.")
                    return False
        print(f"[UPLOAD] Successfully uploaded {filename} to files server!")
        return True
    except Exception as e:
        print(f"[UPLOAD] Error uploading {filename}: {e}")
        return False

# --- VALIDATION FLISTS GENERATION ---
def generate_validation_flists(inpaint_dir, mask_dir):
    # Determine the results directory location (check various nested positions)
    candidates = [
        "TEMP_QUALITATIVE/TEMP_QUALITATIVE/CMT_Validate_Places2/results",
        "TEMP_QUALITATIVE/CMT_Validate_Places2/results",
        "../TEMP_QUALITATIVE/TEMP_QUALITATIVE/CMT_Validate_Places2/results",
        "../TEMP_QUALITATIVE/CMT_Validate_Places2/results",
    ]
    
    results_dir = None
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            results_dir = c
            break
            
    if not results_dir:
        print("[*] Results directory not found. Skipping validation flists generation.")
        return False
        
    print(f"[*] Found qualitative results directory: {results_dir}")
    print("[*] Generating validation flists from image filenames...")
    
    # Try to locate test_256.tar for on-demand extraction of missing files
    tar_candidates = [
        "datasets/test_256.tar",
        "test_256.tar",
        "../datasets/test_256.tar",
        "../test_256.tar"
    ]
    tar_path = None
    for tc in tar_candidates:
        if os.path.exists(tc):
            tar_path = tc
            break

    import tarfile
    tar_handle = None
    if tar_path:
        try:
            tar_handle = tarfile.open(tar_path, 'r')
            print(f"[*] Opened {tar_path} for on-demand extraction of missing images.")
        except Exception as e:
            print(f"[-] Warning: Failed to open {tar_path}: {e}")
            
    categories = ['SMALL', 'MEDIUM', 'LARGE']
    any_generated = False
    
    try:
        for cat in categories:
            cat_dir = os.path.join(results_dir, cat)
            if not os.path.exists(cat_dir) or not os.path.isdir(cat_dir):
                print(f"[-] Subdirectory for category {cat} not found in {results_dir}. Skipping.")
                continue
                
            img_paths = []
            mask_paths = []
            
            # List files in the category directory
            filenames = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            # Sort to ensure deterministic order
            filenames.sort()
            
            for fname in filenames:
                base_name = os.path.splitext(fname)[0]
                # Pattern: [image_id]_[mask_id]_[psnr]
                parts = base_name.rsplit('_', 2)
                if len(parts) != 3:
                    continue
                    
                image_id, mask_id, psnr = parts
                
                # Construct and verify image path
                img_path = None
                for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                    p = os.path.join(inpaint_dir, image_id + ext)
                    if os.path.exists(p):
                        img_path = p
                        break
                        
                # On-demand extraction if file is missing and tar archive is open
                if not img_path and tar_handle:
                    for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                        tar_member_name = f"test_256/{image_id}{ext}"
                        try:
                            # Extract to the directory containing inpaint_dir
                            tar_handle.extract(tar_member_name, path=os.path.dirname(inpaint_dir))
                            p = os.path.join(inpaint_dir, image_id + ext)
                            if os.path.exists(p):
                                img_path = p
                                break
                        except KeyError:
                            continue
                        except Exception as e:
                            print(f"[-] Warning: Failed to extract {tar_member_name}: {e}")
                            break
                            
                if not img_path:
                    p = os.path.join(inpaint_dir, image_id)
                    if os.path.exists(p):
                        img_path = p
                        
                # Construct and verify mask path
                m_path = None
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                    p = os.path.join(mask_dir, mask_id + ext)
                    if os.path.exists(p):
                        m_path = p
                        break
                        
                if img_path and m_path:
                    img_paths.append(os.path.abspath(img_path))
                    mask_paths.append(os.path.abspath(m_path))
                    
            if img_paths:
                images_flist_path = f"datasets/places365/val_images_{cat}.flist"
                masks_flist_path = f"datasets/places365/val_masks_{cat}.flist"
                
                os.makedirs(os.path.dirname(images_flist_path), exist_ok=True)
                
                with open(images_flist_path, 'w') as f:
                    for p in img_paths:
                        f.write(p + '\n')
                        
                with open(masks_flist_path, 'w') as f:
                    for p in mask_paths:
                        f.write(p + '\n')
                        
                print(f"[+] Successfully generated validation flists for {cat}:")
                print(f"    - Images flist ({len(img_paths)} lines): {images_flist_path}")
                print(f"    - Masks flist ({len(mask_paths)} lines): {masks_flist_path}")
                any_generated = True
            else:
                print(f"[-] No valid image-mask pairs found for category {cat} in {cat_dir}.")
    finally:
        if tar_handle:
            tar_handle.close()
            
    return any_generated


# --- MAIN ---
def main():
    print("VERSION: 2.0 - format-aware saving active")
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./PlacesTraining', help='model checkpoints path')
    parser.add_argument('--output', type=str, default='./evaluation_results_images', help='output directory path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for evaluation')
    parser.add_argument('--input-size', type=int, default=256, help='override input image size')
    parser.add_argument('--num-images', type=int, default=2000, help='limit evaluation to first N images')
    args = parser.parse_args()

    config_path = os.path.join(args.path, 'config.yml')
    if not os.path.exists(config_path):
        # Fallback to local config.yml if not found in args.path
        if os.path.exists('./config.yml'):
            print(f"Config not found in {args.path}, copying default ./config.yml...")
            os.makedirs(args.path, exist_ok=True)
            shutil.copyfile('./config.yml', config_path)
        else:
            raise FileNotFoundError(f"Config file not found in {args.path} or root directory.")

    config = Config(config_path)
    config.PATH = args.path
    config.MODE = 2
    config.MODEL = 2
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.WORLD_SIZE = 1
    
    if args.input_size is not None:
        config.INPUT_SIZE = args.input_size
        
    print(f"Evaluation Configurations:")
    print(f"  - Image size: {config.INPUT_SIZE}x{config.INPUT_SIZE}")
    print(f"  - Device: {config.DEVICE}")

    # Resolve dataset paths dynamically
    inpaint_dir = "datasets/places365/test_256"
    mask_dir = "datasets/testing_mask_dataset"
    
    if not os.path.exists(inpaint_dir):
        if os.path.exists("test_256"):
            inpaint_dir = "test_256"
        elif os.path.exists("../test_256"):
            inpaint_dir = "../test_256"
            
    if not os.path.exists(mask_dir):
        if os.path.exists("testing_mask_dataset"):
            mask_dir = "testing_mask_dataset"
        elif os.path.exists("../testing_mask_dataset"):
            mask_dir = "../testing_mask_dataset"

    # Automatically generate validation flists if metrics_LARGE.csv is found
    flist_generated = generate_validation_flists(inpaint_dir, mask_dir)

    # Resolve default/fallback values (which we'll use if flist is not generated)
    default_image_flist = inpaint_dir
    default_mask_flist = mask_dir

    # LPIPS
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)
    loss_fn_vgg.eval()

    # Model
    model = InpaintingModel(config).to(config.DEVICE)
    model.load()
    
    if hasattr(config, 'GPU') and len(config.GPU) > 1 and config.DEVICE.type == 'cuda':
        print(f"Wrapping model with DataParallel on GPUs: {config.GPU}")
        model = nn.DataParallel(model, device_ids=config.GPU)
        
    model.eval()

    # Index Custom Masks (if not using flists)
    indexed_masks = None
    if not flist_generated:
        print("[*] Generating indexes for default custom masks...")
        indexed_masks = index_custom_masks(default_mask_flist)

    categories = ['SMALL', 'MEDIUM', 'LARGE']

    stats = {
        cat: {
            'name': [],
            'psnr': [], 'ssim': [],
            'l1': [], 'lpips': []
        } for cat in categories
    }

    create_dir(args.output)
    
    # Setup temporary dirs for FID to keep the final output zip clean and small
    fid_real_dirs = {cat: os.path.join(args.output, f'temp_fid_real_{cat}') for cat in categories}
    fid_fake_dirs = {cat: os.path.join(args.output, f'temp_fid_fake_{cat}') for cat in categories}
    
    # Output dir for custom named images
    images_output_dirs = {cat: os.path.join(args.output, cat) for cat in categories}

    for cat in categories:
        create_dir(fid_real_dirs[cat])
        create_dir(fid_fake_dirs[cat])
        create_dir(images_output_dirs[cat])

    executor = ThreadPoolExecutor(max_workers=8)

    remaining_images = {}
    for cat in categories:
        csv_path = os.path.join(args.output, f'metrics_{cat}.csv')
        completed_count = 0
        
        # Calculate expected total for this category
        if flist_generated:
            cat_images_flist = f"datasets/places365/val_images_{cat}.flist"
            if os.path.exists(cat_images_flist):
                with open(cat_images_flist, 'r') as f:
                    cat_total = len([line for line in f if line.strip()])
            else:
                cat_total = 0
        else:
            temp_dataset = Dataset(config, default_image_flist, default_mask_flist, augment=False, training=False)
            cat_total = len(temp_dataset.data)
            if args.num_images is not None and cat_total > args.num_images:
                cat_total = args.num_images
                
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                has_average = any(r and r[0] == 'AVERAGE' for r in rows)
                if has_average:
                    completed_count = cat_total
                else:
                    for r in rows[1:]:
                        if r and len(r) >= 5 and r[0].strip() and r[0] != 'Image' and r[0] != 'AVERAGE':
                            completed_count += 1
            except Exception:
                pass
        remaining_images[cat] = max(0, cat_total - completed_count)
    
    total_initial_remaining = sum(remaining_images.values())
    run_start_time = time.time()
    run_processed_count = 0

    send_notification(f"SEM-Net: Image Eval started. Remaining: {total_initial_remaining}")

    # ---------------- LOOP ---------------- #
    for cat in categories:
        csv_path = os.path.join(args.output, f'metrics_{cat}.csv')
        completed_images = set()

        # Check if we can resume from existing CSV
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                
                # Determine expected total for this category
                if flist_generated:
                    cat_images_flist = f"datasets/places365/val_images_{cat}.flist"
                    if os.path.exists(cat_images_flist):
                        with open(cat_images_flist, 'r') as f:
                            cat_total = len([line for line in f if line.strip()])
                    else:
                        cat_total = 0
                else:
                    temp_dataset = Dataset(config, default_image_flist, default_mask_flist, augment=False, training=False)
                    cat_total = len(temp_dataset.data)
                    if args.num_images is not None and cat_total > args.num_images:
                        cat_total = args.num_images
                        
                has_average = any(r and r[0] == 'AVERAGE' for r in rows)
                if has_average:
                    print(f"\nCategory {cat} is already fully completed. Skipping.")
                    for r in rows:
                        if r and r[0] == 'AVERAGE':
                            try:
                                stats[cat]['psnr'] = [float(r[1])]
                                stats[cat]['ssim'] = [float(r[2])]
                                stats[cat]['l1'] = [float(r[3])]
                                stats[cat]['lpips'] = [float(r[4])]
                            except Exception:
                                pass
                    continue
                else:
                    print(f"\nFound partially completed CSV for {cat}. Loading existing progress...")
                    for r in rows[1:]:
                        if r and len(r) >= 5 and r[0].strip() and r[0] != 'Image' and r[0] != 'AVERAGE':
                            img_name = r[0]
                            stats[cat]['name'].append(img_name)
                            stats[cat]['psnr'].append(float(r[1]))
                            stats[cat]['ssim'].append(float(r[2]))
                            stats[cat]['l1'].append(float(r[3]))
                            stats[cat]['lpips'].append(float(r[4]))
                            completed_images.add(img_name)
                    print(f"Loaded {len(completed_images)} completed images for {cat} from CSV.")
            except Exception as e:
                print(f"WARNING: Failed to parse existing CSV for {cat}: {e}. Starting fresh.")
                stats[cat] = {'name': [], 'psnr': [], 'ssim': [], 'l1': [], 'lpips': []}
                completed_images = set()

        print(f"\nEvaluating {cat}")
        
        # Initialize category-specific dataset and loader
        if flist_generated:
            cat_images_flist = f"datasets/places365/val_images_{cat}.flist"
            cat_masks_flist = f"datasets/places365/val_masks_{cat}.flist"
            if not os.path.exists(cat_images_flist) or not os.path.exists(cat_masks_flist):
                print(f"[-] Flist files for category {cat} do not exist. Skipping.")
                continue
                
            config.TEST_INPAINT_IMAGE_FLIST = cat_images_flist
            config.TEST_MASK_FLIST = cat_masks_flist
            config.MASK = 6 # Force dataset to load mask directly from mask_flist at same index
        else:
            config.TEST_INPAINT_IMAGE_FLIST = default_image_flist
            config.TEST_MASK_FLIST = default_mask_flist
            config.MASK = 3
            
        test_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST, config.TEST_MASK_FLIST,
                               augment=False, training=False)
        
        num_total = len(test_dataset.data)
        if num_total == 0:
            print(f"[-] No test images found for category {cat}. Skipping.")
            continue
            
        if args.num_images is not None and num_total > args.num_images:
            indices = np.linspace(0, num_total - 1, args.num_images, dtype=int).tolist()
            test_dataset.data = [test_dataset.data[idx] for idx in indices]
            if hasattr(test_dataset, 'mask_data') and len(test_dataset.mask_data) > 0:
                test_dataset.mask_data = [test_dataset.mask_data[idx] for idx in indices]
            print(f"Dataset limited to {len(test_dataset)} evenly spaced images for category {cat}.")
        else:
            print(f"Dataset has {num_total} images for category {cat}.")
            
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        futures = []
        last_notified_count = (len(stats[cat]['psnr']) // 100) * 100

        for index, items in enumerate(test_loader):
            images, masks_batch = items
            curr_batch_size = images.shape[0]

            # Determine filenames in current batch
            batch_filenames = [test_dataset.load_name(index * args.batch_size + i) for i in range(curr_batch_size)]
            
            # Identify indices in this batch that have not been evaluated yet
            indices_to_evaluate = [i for i, name in enumerate(batch_filenames) if name not in completed_images]
            
            if not indices_to_evaluate:
                continue

            images = images.to(config.DEVICE)
            h, w = images.shape[2], images.shape[3]
            
            if flist_generated:
                masks = masks_batch.to(config.DEVICE)
            else:
                # Enforce stride=2 for even-only mask selection
                masks = get_custom_mask(indexed_masks, cat, index, h, w, curr_batch_size, stride=2).to(config.DEVICE)

            with torch.no_grad():
                outputs_img = model(images, masks)

            outputs_merged = (outputs_img * masks) + (images * (1 - masks))

            with torch.no_grad():
                lpips_batch = loss_fn_vgg(
                    normalize_lpips(outputs_merged),
                    normalize_lpips(images)
                )
                lpips_vals = lpips_batch.flatten().cpu().tolist()

            # -------- METRICS & IMAGE SAVING -------- #
            for i in indices_to_evaluate:
                file_name = batch_filenames[i]
                global_idx = index * args.batch_size + i

                psnr, ssim = calc_psnr_ssim(images[i:i+1], outputs_merged[i:i+1])
                l1_val = torch.nn.functional.l1_loss(outputs_merged[i:i+1], images[i:i+1], reduction='mean').item()
                lpips_val = lpips_vals[i]

                stats[cat]['name'].append(file_name)
                stats[cat]['psnr'].append(psnr)
                stats[cat]['ssim'].append(ssim)
                stats[cat]['l1'].append(l1_val)
                stats[cat]['lpips'].append(lpips_val)
                completed_images.add(file_name)

                # --- IMAGE SAVING ---
                gt_img_pil = Image.fromarray(postprocess(images[i:i+1])[0].cpu().numpy().astype(np.uint8))
                pred_merged_pil = Image.fromarray(postprocess(outputs_merged[i:i+1])[0].cpu().numpy().astype(np.uint8))
                
                # Asynchronously save temporary FID images (with original name)
                futures.append(executor.submit(save_task, os.path.join(fid_real_dirs[cat], file_name), gt_img_pil))
                futures.append(executor.submit(save_task, os.path.join(fid_fake_dirs[cat], file_name), pred_merged_pil))

                # Save custom-named output image: [GT-image-id]_[mask-id]_[PSNR].png
                gt_image_id = os.path.splitext(file_name)[0]
                if flist_generated:
                    mask_path = test_dataset.mask_data[global_idx]
                else:
                    mask_idx = (global_idx * 2) % len(indexed_masks[cat])
                    mask_path = indexed_masks[cat][mask_idx]
                mask_id = os.path.splitext(os.path.basename(mask_path))[0]
                
                custom_save_name = f"{gt_image_id}_{mask_id}_{psnr:.2f}.png"
                futures.append(executor.submit(save_task, os.path.join(images_output_dirs[cat], custom_save_name), pred_merged_pil))

            run_processed_count += len(indices_to_evaluate)

            # Milestone Notification Check
            current_count = len(stats[cat]['psnr'])
            should_notify = False
            
            if current_count - last_notified_count >= 100:
                should_notify = True
                last_notified_count = (current_count // 100) * 100
                
            if should_notify:
                avg_psnr = np.mean(stats[cat]['psnr'])
                elapsed = time.time() - run_start_time
                if run_processed_count > 0:
                    img_per_sec = run_processed_count / elapsed
                    remaining_images_run = max(0, total_initial_remaining - run_processed_count)
                    remaining_time_sec = remaining_images_run / img_per_sec
                    
                    # Adapt ETA to show days
                    days = int(remaining_time_sec // 86400)
                    hours = int((remaining_time_sec % 86400) // 3600)
                    minutes = int((remaining_time_sec % 3600) // 60)
                    seconds = int(remaining_time_sec % 60)
                    if days > 0:
                        eta_str = f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
                    else:
                        eta_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    
                    # Estimate time of finish (in IST timezone, UTC+5:30)
                    from datetime import datetime, timedelta, timezone
                    ist_tz = timezone(timedelta(hours=5, minutes=30))
                    finish_time = datetime.now(timezone.utc).astimezone(ist_tz) + timedelta(seconds=remaining_time_sec)
                    finish_str = finish_time.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    eta_str = "Calculating..."
                    finish_str = "Calculating..."
                    
                send_notification(f"[{cat}] {current_count}/{len(test_dataset)} | PSNR: {avg_psnr:.2f} | ETA: {eta_str} | Finish (IST): {finish_str}")

            # Periodic Incremental Save
            if len(stats[cat]['name']) > 0:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Image', 'PSNR', 'SSIM', 'L1', 'LPIPS'])
                    for idx in range(len(stats[cat]['name'])):
                        writer.writerow([
                            stats[cat]['name'][idx],
                            stats[cat]['psnr'][idx],
                            stats[cat]['ssim'][idx],
                            stats[cat]['l1'][idx],
                            stats[cat]['lpips'][idx]
                        ])

        # Wait for all background image writes to finish before computing FID
        if futures:
            print(f"Waiting for {len(futures)} image writes to complete for {cat}...")
            concurrent.futures.wait(futures)
            print("All writes complete. Computing FID...")

        # Compute FID score using cleanfid
        fid_score = fid.compute_fid(fid_real_dirs[cat], fid_fake_dirs[cat])
        fid_str = f"FID: {fid_score:.4f}"

        # Save Final CSV for this category
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'PSNR', 'SSIM', 'L1', 'LPIPS'])

            for idx in range(len(stats[cat]['name'])):
                writer.writerow([
                    stats[cat]['name'][idx],
                    stats[cat]['psnr'][idx],
                    stats[cat]['ssim'][idx],
                    stats[cat]['l1'][idx],
                    stats[cat]['lpips'][idx]
                ])

            writer.writerow([])

            # Averages
            avg_psnr = np.mean(stats[cat]['psnr'])
            avg_ssim = np.mean(stats[cat]['ssim'])
            avg_l1 = np.mean(stats[cat]['l1'])
            avg_lpips = np.mean(stats[cat]['lpips'])

            writer.writerow([
                'AVERAGE',
                avg_psnr, avg_ssim, avg_l1, avg_lpips,
                fid_str
            ])

        print(f"{cat} done. {fid_str}")

        # Clean up temporary FID directories immediately to free up disk space and avoid zipping them
        try:
            print(f"Cleaning up temporary FID directories for {cat}...")
            shutil.rmtree(fid_real_dirs[cat], ignore_errors=True)
            shutil.rmtree(fid_fake_dirs[cat], ignore_errors=True)
        except Exception as e:
            print(f"Warning: failed to clean up FID directories: {e}")

        # Send category-end notification
        cat_message = (
            f"[{cat}] Done.\n"
            f"PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.3f} | LPIPS: {avg_lpips:.3f} | {fid_str}"
        )
        send_notification(cat_message)

    # Shutdown the thread pool executor
    executor.shutdown(wait=True)
    print("Image generation and metrics computation complete!")
    
    # ---------------- ZIP & UPLOAD ---------------- #
    folder_name = os.path.basename(os.path.normpath(args.output))
    parent_dir = os.path.dirname(os.path.abspath(args.output))
    zip_base_path = os.path.join(parent_dir, folder_name)
    zip_file_path = f"{zip_base_path}.zip"
    
    print(f"Creating zip file: {zip_file_path}...")
    try:
        shutil.make_archive(zip_base_path, 'zip', root_dir=parent_dir, base_dir=folder_name)
        print("Zip created successfully.")
    except Exception as e:
        print(f"Failed to create zip: {e}")
        send_notification(f"SEM-Net: Zipping failed. Error: {e}")
        return

    # Upload to server
    server_url = os.environ.get("FILES_SERVER_URL", "https://files.lalithadithyan.dev")
    session_id = os.environ.get("C2_SESSION", "default")
    
    upload_success = upload_file_chunked(zip_file_path, server_url, session_id)
    
    # Clean up zip file
    if os.path.exists(zip_file_path):
        try:
            os.remove(zip_file_path)
            print("Temporary zip file cleaned up.")
        except Exception as e:
            print(f"Failed to remove temporary zip: {e}")

    final_summary = "SEM-Net: Image evaluation and upload complete.\n"
    if upload_success:
        final_summary += f"Result uploaded to files server under session '{session_id}'.\n"
    else:
        final_summary += "Result upload FAILED.\n"
        
    for c in categories:
        if stats[c]['psnr']:
            avg_psnr = np.mean(stats[c]['psnr'])
            avg_ssim = np.mean(stats[c]['ssim'])
            final_summary += f"{c}: PSNR {avg_psnr:.2f}, SSIM {avg_ssim:.3f}\n"
        else:
            final_summary += f"{c}: No data\n"
            
    send_notification(final_summary)

if __name__ == '__main__':
    main()
