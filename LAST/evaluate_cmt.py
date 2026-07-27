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

# Try to find CMT root path dynamically to handle different folders (local vs server)
cmt_paths = [
    os.path.abspath('TEMP_QUALITATIVE/TEMP_QUALITATIVE/CMT'),
    os.path.abspath('TEMP_QUALITATIVE/CMT'),
    os.path.abspath('CMT')
]
cmt_root = None
for p in cmt_paths:
    if os.path.exists(os.path.join(p, 'network', 'network_pro.py')):
        cmt_root = p
        break

if cmt_root is None:
    print("[WARNING] Could not find CMT source code directory. Defaulting to 'TEMP_QUALITATIVE/TEMP_QUALITATIVE/CMT'")
    cmt_root = os.path.abspath('TEMP_QUALITATIVE/TEMP_QUALITATIVE/CMT')

sys.path.insert(0, cmt_root)

try:
    from network.network_pro import Inpaint
    from utils import load_checkpoint
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[ERROR] Could not import CMT: {e}")
    sys.exit(1)

# GDrive IDs for downloading CMT weights
CMT_WEIGHTS_GDRIVE = {
    'place2': '1zLkKixPnuoAY1k4fdq6JjidlRafKMhXN',
    'celeba': '1e6EbwGnMGgGXAn4QLffT_Zx_BbidBSbR'
}

# Compatibility for older/newer Pillow versions
try:
    LANCZOS = Image.Resampling.LANCZOS
    NEAREST = Image.Resampling.NEAREST
except AttributeError:
    LANCZOS = Image.LANCZOS
    NEAREST = Image.NEAREST

def download_file_from_google_drive(file_id, dest_path):
    import re
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    url = f"https://docs.google.com/uc?export=download&id={file_id}"
    
    try:
        # 1. Fetch Google Drive warning page
        response = session.get(url, headers=headers, timeout=60)
        if response.status_code != 200:
            print(f"[ERROR] Failed to fetch GDrive warning page: HTTP {response.status_code}")
            return False
            
        # Parse inputs using regex
        inputs = re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)">', response.text)
        if not inputs:
            inputs = re.findall(r'<input type="hidden" name=\'([^\']+)\' value=\'([^\']*)\'>', response.text)
        if not inputs:
            inputs = re.findall(r'name="([^"]+)"\s+value="([^"]*)"', response.text)
            
        params = {name: val for name, val in inputs if name in ['id', 'export', 'confirm', 'uuid']}
        
        # Parse action URL using regex
        action_match = re.search(r'action="([^"]+)"', response.text)
        action = action_match.group(1) if action_match else "https://drive.usercontent.google.com/download"
        
        # 2. If confirmation parameters parsed, download file using User-Agent session
        if params:
            print("[*] GDrive warning form parsed successfully. Starting download...")
            download_res = session.get(action, params=params, headers=headers, stream=True, timeout=120)
        else:
            print("[*] No GDrive confirmation warning form parsed. Attempting direct download...")
            download_res = session.get(url, headers=headers, stream=True, timeout=120)
            
        if download_res.status_code == 200:
            content_type = download_res.headers.get('content-type', '').lower()
            if 'html' in content_type:
                print("[ERROR] Google Drive returned an HTML page instead of the binary model file.")
                return False
                
            total_size = int(download_res.headers.get('content-length', 0))
            with open(dest_path, 'wb') as f, tqdm(
                desc=os.path.basename(dest_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in download_res.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            return True
        else:
            print(f"[ERROR] GDrive download request failed: HTTP {download_res.status_code}")
    except Exception as e:
        print(f"[ERROR] Google Drive download failed: {e}")
    return False


def upload_file_chunked(file_path, server_url, session_id, chunk_size=10 * 1024 * 1024):
    if not os.path.exists(file_path):
        print(f"[UPLOAD] File {file_path} not found.")
        return False
        
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    total_chunks = (file_size + chunk_size - 1) // chunk_size
    
    print(f"[UPLOAD] Uploading {filename} ({file_size / (1024*1024):.2f} MB) in {total_chunks} chunks...")
    
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
                        res = requests.post(f"{server_url}/api/upload_chunk", files=files, data=data, timeout=45)
                        print(f"  -> Chunk {i}: Server status {res.status_code}")
                        if res.status_code == 200:
                            success = True
                            break
                    except Exception as e:
                        print(f"  -> Chunk {i} retry {retry+1} error: {e}")
                    time.sleep(1)
                    
                if not success:
                    print(f"[UPLOAD] Failed to upload chunk {i}.")
                    return False
        print(f"[UPLOAD] Successfully uploaded {filename} to files server!")
        return True
    except Exception as e:
        print(f"[UPLOAD] Error uploading {filename}: {e}")
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
    if os.path.isdir(model_path):
        pth_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
        if pth_files:
            checkpoint_path = os.path.join(model_path, pth_files[0])
        else:
            raise FileNotFoundError(f"No checkpoint file found in directory: {model_path}")
    else:
        checkpoint_path = model_path
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = Inpaint()
    model = load_checkpoint(checkpoint_path, model)
    model.to(device)
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
    try:
        topic = "camino-places-eval-2006"
        requests.post(f"https://ntfy.sh/{topic}", data=message.encode(encoding='utf-8'), timeout=5)
        print(f"Sent push notification: {message}")
    except Exception as e:
        print(f"Failed to send push notification: {e}")

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
    if os.path.exists(model_path):
        # Verify if the existing file is a corrupted HTML file (starts with '<')
        try:
            with open(model_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_chars = f.read(10)
                if first_chars.startswith('<'):
                    print(f"[!] Warning: {model_path} appears to be a corrupted HTML file. Deleting and re-downloading...")
                    os.remove(model_path)
                else:
                    return
        except Exception:
            return
        
    filename = os.path.basename(model_path).lower()
    
    dir_name = os.path.dirname(model_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    file_id = None
    if 'place' in filename or 'places' in filename:
        file_id = CMT_WEIGHTS_GDRIVE['place2']
        model_name = "Places2"
    elif 'celeba' in filename:
        file_id = CMT_WEIGHTS_GDRIVE['celeba']
        model_name = "CelebA"
    else:
        file_id = CMT_WEIGHTS_GDRIVE['place2']
        model_name = "Places2"
        
    print(f"[*] CMT weights not found at {model_path}. Downloading {model_name} weights from Google Drive...")
    success = download_file_from_google_drive(file_id, model_path)
    if success:
        print(f"[+] Downloaded {model_name} weights successfully to {model_path}")
    else:
        print(f"[ERROR] Failed to download CMT weights from Google Drive.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate CMT on CelebA-HQ 256 test dataset with custom strided masks")
    parser.add_argument('--model-path', type=str, default='TEMP_QUALITATIVE/TEMP_QUALITATIVE/CMT_Validate_Places2/Place2.pth', help='Path to CMT model checkpoint or directory')
    parser.add_argument('--image-dir', type=str, default='datasets/celeba_hq_256_test', help='Path to test images')
    parser.add_argument('--mask-dir', type=str, default='datasets/testing_mask_dataset', help='Path to testing masks')
    parser.add_argument('--output-dir', type=str, default='evaluation_results_cmt', help='Output directory')
    parser.add_argument('--num-images', type=int, default=2000, help='Number of test images to evaluate')
    parser.add_argument('--log-file', type=str, default='cmt_evaluation.log', help='Path to save terminal logs')
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
    send_notification(f"CMT Eval: Started. Total images: {num_images} on device: {device}")

    eval_start_time = time.time()
    category_summaries = []  # collect (cat, avg_psnr, min_psnr, max_psnr, count)

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

        # ── Resume: scan already-done outputs ────────────────────────
        cat_out_dir = os.path.join(args.output_dir, cat)
        os.makedirs(cat_out_dir, exist_ok=True)
        done_keys = {}  # key: (img_id, msk_id)  ->  psnr_val (float)
        for fname in os.listdir(cat_out_dir):
            if not fname.endswith('.png'):
                continue
            # expected pattern: {cat}_im{img_id}_mask{msk_id}_{psnr}.png
            try:
                # strip category prefix and extension
                body = fname[len(cat) + 3:]          # remove "{cat}_im"
                mask_part = body.split('_mask')
                img_id_done = mask_part[0]
                rest = mask_part[1]                  # "{msk_id}_{psnr}.png"
                # last underscore separates msk_id from psnr
                last_us = rest.rfind('_')
                msk_id_done = rest[:last_us]
                psnr_done = float(rest[last_us + 1:-4])  # strip .png
                done_keys[(img_id_done, msk_id_done)] = psnr_done
                psnr_list.append(psnr_done)
            except Exception:
                pass  # ignore files that don't match the pattern

        skipped = len(done_keys)
        if skipped > 0:
            print(f"  [RESUME] Found {skipped} already-done images for '{cat}'. Skipping them.")

        bar_fmt = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}  PSNR={postfix[psnr]:.2f} avg={postfix[avg]:.2f}]"
        )
        avg_resume = float(np.mean(psnr_list)) if psnr_list else 0.0
        pbar = tqdm(
            total=num_images,
            initial=skipped,          # bar starts at already-done count
            desc=f"[{cat}]",
            unit="img",
            dynamic_ncols=True,
            postfix={"psnr": avg_resume, "avg": avg_resume},
        )

        for idx in range(num_images):
            img_name = image_files[idx]
            img_path = os.path.join(args.image_dir, img_name)

            # Stride 2 mask selection
            mask_idx = (idx * 2) % len(cat_masks)
            mask_path = cat_masks[mask_idx]
            mask_name = os.path.basename(mask_path)

            # ── Resume: skip already-done pairs ──────────────────────
            img_id_check = os.path.splitext(img_name)[0]
            msk_id_check = os.path.splitext(mask_name)[0]
            if (img_id_check, msk_id_check) in done_keys:
                continue  # already on disk, psnr already in psnr_list

            img_step_start = time.time()

            # Load and preprocess image
            img_pil = Image.open(img_path).convert('RGB')
            # Ensure 256x256 for CMT compatibility
            if img_pil.size != (256, 256):
                img_pil = img_pil.resize((256, 256), LANCZOS)
            w, h = img_pil.size
            img_tensor = torch.from_numpy(np.array(img_pil).transpose(2, 0, 1)).float() / 255.0
            # CMT expects range [-1, 1]
            img_tensor = img_tensor * 2.0 - 1.0

            # Load and preprocess mask
            mask_pil = Image.open(mask_path).convert('L')
            if mask_pil.size != (256, 256):
                mask_pil = mask_pil.resize((256, 256), NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_pil)).float() / 255.0
            mask_tensor = (mask_tensor > 0.5).float().unsqueeze(0)  # 1 x H x W

            # Prepare batch inputs
            img_tensor_batch = img_tensor.unsqueeze(0).to(device)
            mask_tensor_batch = mask_tensor.unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                pred_tensor_batch = model(img_tensor_batch, mask_tensor_batch)
                pred_tensor = pred_tensor_batch[0]

            # Postprocess CMT output to range [0, 1]
            pred_tensor = torch.clamp(pred_tensor, -1.0, 1.0) * 0.5 + 0.5
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
            mask_gray_3ch = np.stack([mask_np] * 3, axis=-1)
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

            # Timing
            img_elapsed = time.time() - img_step_start
            avg_psnr = float(np.mean(psnr_list))

            # Update tqdm bar with live PSNR stats
            pbar.set_postfix({"psnr": psnr_val, "avg": avg_psnr}, refresh=False)
            pbar.update(1)

            # Verbose per-image log every 10 images
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

            # Mobile notification after every 500 images
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

    # ── Final Summary Table ───────────────────────────────────────────
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
            f"CMT Eval DONE in {datetime.timedelta(seconds=int(total_elapsed))}. "
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
            
            # Upload the zipped file
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
