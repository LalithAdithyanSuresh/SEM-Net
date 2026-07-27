import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
import requests
from tqdm import tqdm
import time
import datetime
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# Try to find HINT root path dynamically to handle different folders (local vs server)
hint_paths = [
    os.path.abspath('TEMP_QUALITATIVE/TEMP_QUALITATIVE/HINT'),
    os.path.abspath('TEMP_QUALITATIVE/HINT'),
    os.path.abspath('HINT')
]
hint_root = None
for p in hint_paths:
    if os.path.exists(os.path.join(p, 'src', 'models.py')):
        hint_root = p
        break

if hint_root is None:
    print("[WARNING] Could not find HINT source code directory. Defaulting to 'TEMP_QUALITATIVE/TEMP_QUALITATIVE/HINT'")
    hint_root = os.path.abspath('TEMP_QUALITATIVE/TEMP_QUALITATIVE/HINT')

sys.path.insert(0, hint_root)

try:
    from src.config import Config
    from src.models import InpaintingModel
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[ERROR] Could not import HINT: {e}")
    sys.exit(1)

# Compatibility for older/newer Pillow versions
try:
    LANCZOS = Image.Resampling.LANCZOS
    NEAREST = Image.Resampling.NEAREST
except AttributeError:
    LANCZOS = Image.LANCZOS
    NEAREST = Image.NEAREST

def upload_file_chunked(file_path, server_url, session_id, chunk_size=10 * 1024 * 1024):
    return False

class Tee:
    def __init__(self, original_stream, file_handle):
        self.original_stream = original_stream
        self.file_handle = file_handle

    def write(self, message):
        self.original_stream.write(message)
        self.file_handle.write(message)
        self.file_handle.flush()

    def flush(self):
        self.original_stream.flush()
        self.file_handle.flush()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)

def load_model(model_path, device):
    if not os.path.isdir(model_path):
        if os.path.basename(model_path) in ['InpaintingModel_gen.pth', 'InpaintingModel_dis.pth']:
            model_path = os.path.dirname(model_path)
        else:
            raise FileNotFoundError(f"Model path must be a directory containing config.yml: {model_path}")
            
    cfg_path = os.path.join(model_path, 'config.yml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"HINT configuration file config.yml not found in: {model_path}")
        
    print(f"Loading HINT model from: {model_path}")
    config = Config(cfg_path)
    config._dict['PATH']     = model_path
    config._dict['MODE']     = 2
    config._dict['MODEL']    = 2
    config._dict['GPU']      = [0]
    config._dict['GAN_LOSS'] = 'lsgan'
    config._dict['DEVICE']   = device
    
    model = InpaintingModel(config).to(device)
    model.load()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def index_custom_masks(mask_dir):
    print(f"Indexing masks in {mask_dir}...")
    categories = {'SMALL': [], 'MEDIUM': [], 'LARGE': []}
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    mask_files.sort()
    
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
            print(f"Error loading mask {f}: {e}")
            
    print(f"Index complete: SMALL({len(categories['SMALL'])}), MEDIUM({len(categories['MEDIUM'])}), LARGE({len(categories['LARGE'])})")
    return categories

def get_image_files(image_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    return image_files

def send_notification(message):
    print(f"[LOCAL NOTIFICATION] {message}")

def ensure_dataset(dest_dir, url, name):
    parent_dir = os.path.dirname(os.path.abspath(dest_dir))
    os.makedirs(parent_dir, exist_ok=True)
    zip_path = os.path.join(parent_dir, f"{name}.zip")
    print(f"[*] Missing dataset at {dest_dir}. Downloading {name} from {url}...")
    try:
        import zipfile
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(
            desc=name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"[*] Extracting {zip_path} to {parent_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(parent_dir)
        os.remove(zip_path)
        print(f"[*] {name} dataset downloaded and extracted successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download {name}: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)

def ensure_model(model_path):
    if os.path.exists(model_path) and os.path.isdir(model_path):
        gen_path = os.path.join(model_path, 'InpaintingModel_gen.pth')
        if os.path.exists(gen_path):
            try:
                with open(gen_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_chars = f.read(10)
                    if first_chars.startswith('<'):
                        print(f"[!] Warning: {gen_path} appears to be a corrupted HTML file. Deleting...")
                        os.remove(gen_path)
            except Exception:
                pass

    if not os.path.exists(model_path) or not os.path.isdir(model_path):
        print(f"[ERROR] HINT checkpoint directory not found at: {model_path}")
        print("Please download the HINT pretrained weights and config.yml, place them in a directory, and pass the path using --model-path.")
        sys.exit(1)
        
    cfg_path = os.path.join(model_path, 'config.yml')
    gen_path = os.path.join(model_path, 'InpaintingModel_gen.pth')
    if not os.path.exists(cfg_path) or not os.path.exists(gen_path):
        print(f"[ERROR] Missing HINT files inside '{model_path}'. We need both 'config.yml' and 'InpaintingModel_gen.pth'.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate HINT on CelebA-HQ 256 test dataset with custom strided masks")
    parser.add_argument('--model-path', type=str, default='TEMP_QUALITATIVE/TEMP_QUALITATIVE/HINT/HINT_Validate_Places2', help='Path to HINT model directory containing config.yml')
    parser.add_argument('--image-dir', type=str, default='datasets/celeba_hq_256_test', help='Path to test images')
    parser.add_argument('--mask-dir', type=str, default='datasets/testing_mask_dataset', help='Path to testing masks')
    parser.add_argument('--output-dir', type=str, default='evaluation_results_hint', help='Output directory')
    parser.add_argument('--num-images', type=int, default=2000, help='Number of test images to evaluate')
    parser.add_argument('--log-file', type=str, default='hint_evaluation.log', help='Path to save terminal logs')
    args = parser.parse_args()

    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_file_handle = open(args.log_file, 'w', encoding='utf-8')
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(sys.stdout, log_file_handle)
        sys.stderr = Tee(sys.stderr, log_file_handle)
        
        import atexit
        def cleanup():
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file_handle.close()
            
            try:
                server_url = os.environ.get("FILES_SERVER_URL", "https://files.lalithadithyan.dev")
                session_id = os.environ.get("C2_SESSION", "DAVA")
                if os.path.exists(args.log_file) and os.path.getsize(args.log_file) > 0:
                    upload_file_chunked(args.log_file, server_url, session_id)
            except Exception as e:
                original_stderr.write(f"[LOG UPLOAD] Failed to auto-upload logs: {e}\n")
                original_stderr.flush()
        atexit.register(cleanup)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    ensure_model(args.model_path)
    model = load_model(args.model_path, device)
    
    # Get Datasets
    if not os.path.exists(args.mask_dir):
        if args.mask_dir == 'datasets/testing_mask_dataset' or os.path.basename(args.mask_dir) == 'testing_mask_dataset':
            ensure_dataset(args.mask_dir, "https://files.lalithadithyan.dev/download/testing_mask_dataset.zip", "testing_mask_dataset")
            
    if not os.path.exists(args.image_dir):
        if args.image_dir == 'datasets/celeba_hq_256_test' or os.path.basename(args.image_dir) == 'celeba_hq_256_test':
            ensure_dataset(args.image_dir, "https://files.lalithadithyan.dev/download/celeba_hq_256_test.zip", "celeba_hq_256_test")

    indexed_masks = index_custom_masks(args.mask_dir)
    image_files = get_image_files(args.image_dir)
    
    num_images = min(len(image_files), args.num_images)
    image_files = image_files[:num_images]
    print(f"Evaluating first {num_images} images.")
    
    categories = ['SMALL', 'MEDIUM', 'LARGE']
    send_notification(f"HINT Eval: Started. Total images: {num_images} on device: {device}")

    eval_start_time = time.time()
    category_summaries = []

    for cat in categories:
        cat_masks = indexed_masks[cat]
        if not cat_masks:
            print(f"\n[SKIP] No masks found for category '{cat}'. Skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  Category: {cat}  |  Images: {num_images}  |  Masks available: {len(cat_masks)}")
        print(f"{'='*60}")
        os.makedirs(os.path.join(args.output_dir, cat), exist_ok=True)

        psnr_list = []
        cat_start = time.time()

        # ── Resume Scan ────────────────────────
        cat_out_dir = os.path.join(args.output_dir, cat)
        os.makedirs(cat_out_dir, exist_ok=True)
        done_keys = {}
        for fname in os.listdir(cat_out_dir):
            if not fname.endswith('.png'):
                continue
            try:
                body = fname[len(cat) + 3:]
                mask_part = body.split('_mask')
                img_id_done = mask_part[0]
                rest = mask_part[1]
                last_us = rest.rfind('_')
                msk_id_done = rest[:last_us]
                psnr_done = float(rest[last_us + 1:-4])
                done_keys[(img_id_done, msk_id_done)] = psnr_done
                psnr_list.append(psnr_done)
            except Exception:
                pass

        skipped = len(done_keys)
        if skipped > 0:
            print(f"  [RESUME] Found {skipped} already-done images for '{cat}'. Skipping them.")

        avg_resume = float(np.mean(psnr_list)) if psnr_list else 0.0
        pbar = tqdm(
            total=num_images,
            initial=skipped,
            desc=f"[{cat}]",
            unit="img",
            dynamic_ncols=True,
            postfix={"psnr": avg_resume, "avg": avg_resume},
        )

        for idx in range(num_images):
            img_name = image_files[idx]
            img_path = os.path.join(args.image_dir, img_name)

            mask_idx = (idx * 2) % len(cat_masks)
            mask_path = cat_masks[mask_idx]
            mask_name = os.path.basename(mask_path)

            img_id_check = os.path.splitext(img_name)[0]
            msk_id_check = os.path.splitext(mask_name)[0]
            if (img_id_check, msk_id_check) in done_keys:
                continue

            img_step_start = time.time()

            # Load and preprocess image (HINT expects [0, 1])
            img_pil = Image.open(img_path).convert('RGB')
            if img_pil.size != (256, 256):
                img_pil = img_pil.resize((256, 256), LANCZOS)
            img_tensor = torch.from_numpy(np.array(img_pil).transpose(2, 0, 1)).float() / 255.0
            
            # Load and preprocess mask
            mask_pil = Image.open(mask_path).convert('L')
            if mask_pil.size != (256, 256):
                mask_pil = mask_pil.resize((256, 256), NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_pil)).float() / 255.0
            mask_tensor = (mask_tensor > 0.5).float().unsqueeze(0) # 1 x H x W
            
            # Prepare batch inputs
            img_tensor_batch = img_tensor.unsqueeze(0).to(device)
            mask_tensor_batch = mask_tensor.unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                out = model(img_tensor_batch, mask_tensor_batch)
                # Composite
                pred_tensor = (out * mask_tensor_batch + img_tensor_batch * (1 - mask_tensor_batch)).clamp(0, 1)[0]
                
            pred_tensor = pred_tensor.cpu()
            
            # Convert to numpy arrays
            pred_np = (pred_tensor.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            gt_np = np.array(img_pil)
            mask_np = (mask_tensor[0].numpy() * 255.0).astype(np.uint8)
            
            # Compute Merged image
            mask_binary = (mask_tensor[0].numpy() > 0.5).astype(np.uint8)
            mask_binary_3ch = np.expand_dims(mask_binary, axis=-1)
            merged_np = pred_np * mask_binary_3ch + gt_np * (1 - mask_binary_3ch)
            
            # Compute PSNR
            psnr_val = compare_psnr(gt_np, merged_np, data_range=255)
            psnr_list.append(psnr_val)
            
            # Masked Input (white pixels where masked)
            masked_np = gt_np * (1 - mask_binary_3ch) + mask_binary_3ch * 255
            
            # Create 5-image grid
            mask_gray_3ch = np.stack([mask_np]*3, axis=-1)
            grid_np = np.hstack([
                gt_np,
                masked_np,
                mask_gray_3ch,
                pred_np,
                merged_np
            ])
            
            # Filename: <cat>_im<image_id>_mask<mask_id>_<psnr>.png
            img_id = os.path.splitext(img_name)[0]
            msk_id = os.path.splitext(mask_name)[0]
            save_name = f"{cat}_im{img_id}_mask{msk_id}_{psnr_val:.2f}.png"
            save_path = os.path.join(args.output_dir, cat, save_name)
            
            Image.fromarray(grid_np).save(save_path)
            
            img_elapsed = time.time() - img_step_start
            avg_psnr = float(np.mean(psnr_list))
            
            pbar.set_postfix({"psnr": psnr_val, "avg": avg_psnr}, refresh=False)
            pbar.update(1)
            
            if (idx + 1) % 10 == 0 or idx == 0:
                elapsed_total = time.time() - cat_start
                rate = (idx + 1) / elapsed_total if elapsed_total > 0 else 0
                remaining_imgs = num_images - (idx + 1)
                eta_secs = remaining_imgs / rate if rate > 0 else 0
                eta_str = str(datetime.timedelta(seconds=int(eta_secs)))
                print(
                    f"  [{cat}] {idx+1:>5}/{num_images}  "
                    f"img={img_name:<20}  "
                    f"PSNR={psnr_val:6.2f} dB  "
                    f"avg={avg_psnr:6.2f} dB  "
                    f"min={min(psnr_list):6.2f}  max={max(psnr_list):6.2f}  "
                    f"step={img_elapsed*1000:.0f}ms  "
                    f"ETA={eta_str}"
                )
                
            if (idx + 1) % 500 == 0:
                avg_now = float(np.mean(psnr_list))
                send_notification(
                    f"[{cat}] {idx+1}/{num_images} done | "
                    f"PSNR latest={psnr_val:.2f} avg={avg_now:.2f}"
                )
                
        pbar.close()
        
        cat_elapsed = time.time() - cat_start
        avg_psnr_final = float(np.mean(psnr_list)) if psnr_list else 0.0
        min_psnr = float(np.min(psnr_list)) if psnr_list else 0.0
        max_psnr = float(np.max(psnr_list)) if psnr_list else 0.0
        category_summaries.append((cat, avg_psnr_final, min_psnr, max_psnr, len(psnr_list)))
        
        print(f"\n  [{cat}] Done in {datetime.timedelta(seconds=int(cat_elapsed))}  "
              f"| avg PSNR={avg_psnr_final:.4f} dB  "
              f"| min={min_psnr:.2f}  max={max_psnr:.2f}  "
              f"| images={len(psnr_list)}")
              
        send_notification(
            f"[{cat}] Complete! avg PSNR={avg_psnr_final:.2f} dB "
            f"min={min_psnr:.2f} max={max_psnr:.2f} ({len(psnr_list)} imgs)"
        )

    total_elapsed = time.time() - eval_start_time
    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE  |  Total time: {datetime.timedelta(seconds=int(total_elapsed))}")
    print(f"{'='*60}")
    print(f"  {'Category':<10}  {'Images':>7}  {'Avg PSNR':>10}  {'Min PSNR':>10}  {'Max PSNR':>10}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}")
    for cat, avg, mn, mx, cnt in category_summaries:
        print(f"  {cat:<10}  {cnt:>7}  {avg:>10.4f}  {mn:>10.2f}  {mx:>10.2f}")
    print(f"{'='*60}\n")

    if category_summaries:
        all_avg = float(np.mean([x[1] for x in category_summaries]))
        send_notification(
            f"HINT Eval DONE in {datetime.timedelta(seconds=int(total_elapsed))}. "
            f"Overall avg PSNR={all_avg:.2f} dB"
        )

    # ── Zip and Upload Results ────────────────────────────────────────
    if os.path.exists(args.output_dir):
        print(f"\n[*] Zipping results directory '{args.output_dir}'...")
        zip_name = f"{args.output_dir}.zip"
        try:
            import shutil
            shutil.make_archive(args.output_dir, 'zip', args.output_dir)
            print(f"[+] Zip created successfully: {zip_name} ({os.path.getsize(zip_name) / (1024*1024):.2f} MB)")
            
            # Upload the zipped file to validation server
            server_url = os.environ.get("VALIDATION_SERVER_URL", "https://validate.lalithadithyan.dev")
            session_id = os.environ.get("C2_SESSION", "DAVA")
            print(f"[*] Uploading zip file to validation server '{server_url}'...")
            upload_success = upload_file_chunked(zip_name, server_url, session_id)
            if upload_success:
                print(f"[+] Results zip successfully uploaded!")
                try:
                    os.remove(zip_name)
                    print(f"[*] Cleaned up local zip file: {zip_name}")
                except Exception:
                    pass
            else:
                print(f"[ERROR] Failed to upload results zip file.")
        except Exception as e:
            print(f"[ERROR] Failed to zip/upload results: {e}")

if __name__ == '__main__':
    main()
