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
    print(f"Indexing masks in {mask_dir} by numeric filenames...")
    categories = {'SMALL': [], 'MEDIUM': [], 'LARGE': []}
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort files numerically
    mask_files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 999999)
    
    for f in tqdm(mask_files):
        mask_path = os.path.join(mask_dir, f)
        base = os.path.splitext(f)[0]
        if base.isdigit():
            val = int(base)
            if 0 <= val < 4000:
                categories['SMALL'].append(mask_path)
            elif 4000 <= val < 8000:
                categories['MEDIUM'].append(mask_path)
            elif 8000 <= val < 12000:
                categories['LARGE'].append(mask_path)
        else:
            # Fallback to ratio check if not numeric
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

# --- MAIN ---
def main():
    print("VERSION: 2.0 - format-aware saving active")
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./PlacesTraining', help='model checkpoints path')
    parser.add_argument('--output', type=str, default='./evaluation_results_images', help='output directory path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for evaluation')
    parser.add_argument('--input-size', type=int, default=256, help='override input image size')
    parser.add_argument('--num-images', type=int, default=2000, help='limit evaluation to first N images')
    parser.add_argument('--limit-eval', type=int, default=20, help='limit evaluation to the first N images (e.g. 20) for quick test')
    parser.add_argument('--ignore-flists', action='store_true', help='ignore val_images_*.flist files even if they exist')
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
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Limit images: {args.num_images}")
    print(f"  - Device: {config.DEVICE}")
    # Set relative dataset paths for Places365 testing
    config.TEST_INPAINT_IMAGE_FLIST = "datasets/places365/test_256"
    config.TEST_MASK_FLIST = "datasets/testing_mask_dataset"

    print(f"  - Test images directory: {config.TEST_INPAINT_IMAGE_FLIST}")
    print(f"  - Test masks directory: {config.TEST_MASK_FLIST}")

    test_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST, config.TEST_MASK_FLIST,
                           augment=False, training=False)
    
    num_total = len(test_dataset.data)
    if num_total == 0:
        raise ValueError(f"No test images found in directory: {config.TEST_INPAINT_IMAGE_FLIST}. Please verify your config.yml or path.")
        
    if args.num_images is not None and num_total > args.num_images:
        indices = np.linspace(0, num_total - 1, args.num_images, dtype=int).tolist()
        test_dataset.data = [test_dataset.data[idx] for idx in indices]
        print(f"Dataset limited to {len(test_dataset)} evenly spaced images sampled from the total {num_total}.")
    else:
        print(f"Dataset has {num_total} images (using all).")

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
 
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Index Custom Masks
    mask_dir = config.TEST_MASK_FLIST
    indexed_masks = index_custom_masks(mask_dir)

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
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                has_average = any(r and r[0] == 'AVERAGE' for r in rows)
                if has_average:
                    completed_count = len(test_dataset)
                else:
                    for r in rows[1:]:
                        if r and len(r) >= 5 and r[0].strip() and r[0] != 'Image' and r[0] != 'AVERAGE':
                            completed_count += 1
            except Exception:
                pass
        remaining_images[cat] = max(0, len(test_dataset) - completed_count)
    
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
        futures = []
        last_notified_count = (len(stats[cat]['psnr']) // 100) * 100

        # Check for category-specific flists
        img_flist = f"datasets/places365/val_images_{cat}.flist"
        mask_flist = f"datasets/places365/val_masks_{cat}.flist"
        
        if not args.ignore_flists and os.path.exists(img_flist) and os.path.exists(mask_flist):
            print(f"Loading category-specific flist pairs: {img_flist} and {mask_flist}")
            config.MASK = 6  # Non-random external mask loading
            cat_dataset = Dataset(config, img_flist, mask_flist, augment=False, training=False)
            if args.num_images is not None and len(cat_dataset) > args.num_images:
                cat_dataset.data = cat_dataset.data[:args.num_images]
                cat_dataset.mask_data = cat_dataset.mask_data[:args.num_images]
            print(f"Category {cat} dataset loaded with {len(cat_dataset)} pairs.")
            use_flist = True
        else:
            print(f"Category flists not found (or ignored). Using default directories.")
            config.MASK = 3
            cat_dataset = test_dataset
            use_flist = False
            
        cat_loader = DataLoader(
            cat_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        for index, items in enumerate(cat_loader):
            if use_flist:
                images, masks = items
                images = images.to(config.DEVICE)
                masks = masks.to(config.DEVICE)
            else:
                images, _ = items
                images = images.to(config.DEVICE)
                
            curr_batch_size = images.shape[0]

            # Determine filenames in current batch
            batch_filenames = [cat_dataset.load_name(index * args.batch_size + i) for i in range(curr_batch_size)]
            
            # Identify indices in this batch that have not been evaluated yet
            indices_to_evaluate = [i for i, name in enumerate(batch_filenames) if name not in completed_images]
            
            if not indices_to_evaluate:
                continue

            h, w = images.shape[2], images.shape[3]
            
            if not use_flist:
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
                if use_flist:
                    mask_path = cat_dataset.mask_data[global_idx]
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
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time_sec))
                else:
                    eta_str = "Calculating..."
                send_notification(f"[{cat}] {current_count}/{len(test_dataset)} | PSNR: {avg_psnr:.2f} | ETA: {eta_str}")

            # Early evaluation limit termination check
            if args.limit_eval is not None and args.limit_eval > 0 and len(stats[cat]['psnr']) >= args.limit_eval:
                print(f"Reached evaluation limit of {args.limit_eval} images for {cat}. Stopping evaluation loop.")
                break

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
    session_id = os.environ.get("C2_SESSION", "DAVA")
    
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
