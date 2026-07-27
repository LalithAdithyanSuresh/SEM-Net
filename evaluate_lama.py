import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
import requests
from tqdm import tqdm
import time
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# Add the cloned LaMa folder to the Python path
sys.path.insert(0, os.path.abspath('lama'))

try:
    from saicinpainting.training.trainers import load_checkpoint
    from omegaconf import OmegaConf
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[ERROR] Could not import saicinpainting or omegaconf: {e}")
    sys.exit(1)

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
    config_path = os.path.join(model_path, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    # Ensure TORCH_HOME is defined in the environment so that LaMa knows where to load or save weights
    if 'TORCH_HOME' not in os.environ:
        os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
        
    # Check and download missing ADE20K resnet50 dilated encoder weights required for perceptual loss
    torch_home = os.environ['TORCH_HOME']
    ade_dir = os.path.join(torch_home, 'ade20k', 'ade20k-resnet50dilated-ppm_deepsup')
    ade_weight_path = os.path.join(ade_dir, 'encoder_epoch_20.pth')
    
    if not os.path.exists(ade_weight_path):
        print(f"[*] Downloading missing ADE20K resnet50 dilated encoder weights to {ade_weight_path}...")
        os.makedirs(ade_dir, exist_ok=True)
        url = "http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth"
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(ade_weight_path, 'wb') as f, tqdm(
                desc="ADE20K weights",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            print("[*] Download complete!")
        except Exception as e:
            print(f"[ERROR] Failed to download ADE20K weights: {e}")
        
    # Register 'env' resolver for environment variables in configuration
    try:
        if not OmegaConf.has_resolver('env'):
            OmegaConf.register_new_resolver('env', lambda var, default='': os.getenv(var, default))
    except Exception:
        try:
            OmegaConf.register_new_resolver('env', lambda var, default='': os.getenv(var, default), replace=True)
        except Exception:
            pass

    with open(config_path, 'r') as f:
        train_config = OmegaConf.load(f)
    
    # Fix any hardcoded paths pointing to /group-volume
    from omegaconf import DictConfig, ListConfig
    def fix_config_paths(cfg):
        if isinstance(cfg, DictConfig):
            for k, v in cfg.items():
                if isinstance(v, str) and (v.startswith('/group-volume') or v.startswith('/group_volume')):
                    cfg[k] = v.replace('/group-volume', './group-volume').replace('/group_volume', './group_volume')
                elif isinstance(v, (DictConfig, ListConfig)):
                    fix_config_paths(v)
        elif isinstance(cfg, ListConfig):
            for i, v in enumerate(cfg):
                if isinstance(v, str) and (v.startswith('/group-volume') or v.startswith('/group_volume')):
                    cfg[i] = v.replace('/group-volume', './group-volume').replace('/group_volume', './group_volume')
                elif isinstance(v, (DictConfig, ListConfig)):
                    fix_config_paths(v)
                    
    fix_config_paths(train_config)
    
    # Locate the checkpoint file
    models_dir = os.path.join(model_path, 'models')
    checkpoint_path = os.path.join(models_dir, 'best.ckpt')
    if not os.path.exists(checkpoint_path):
        # Fallback to look for any .pt or .ckpt file inside the models directory
        ckpt_files = [f for f in os.listdir(models_dir) if f.endswith(('.pt', '.ckpt'))]
        if ckpt_files:
            checkpoint_path = os.path.join(models_dir, ckpt_files[0])
        else:
            raise FileNotFoundError(f"No checkpoint file found in: {models_dir}")
            
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
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
    if not os.path.exists(model_path) and (model_path == 'lama/lama-fourier-celeba' or os.path.basename(model_path) == 'lama-fourier-celeba'):
        print(f"[*] Missing model directory at {model_path}. Downloading CelebA-HQ LaMa weights...")
        os.makedirs(os.path.join(model_path, 'models'), exist_ok=True)
        config_url = "https://huggingface.co/camenduru/big-lama/resolve/main/lama-celeba-hq/lama-fourier/config.yaml"
        ckpt_url = "https://huggingface.co/camenduru/big-lama/resolve/main/lama-celeba-hq/lama-fourier/models/best.ckpt"
        
        # Download config
        try:
            r = requests.get(config_url, timeout=30)
            r.raise_for_status()
            with open(os.path.join(model_path, 'config.yaml'), 'wb') as f:
                f.write(r.content)
            print("[*] config.yaml downloaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to download config.yaml: {e}")
            
        # Download best.ckpt
        try:
            response = requests.get(ckpt_url, stream=True, timeout=120)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            ckpt_path = os.path.join(model_path, 'models', 'best.ckpt')
            with open(ckpt_path, 'wb') as f, tqdm(
                desc="best.ckpt",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            print("[*] best.ckpt downloaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to download best.ckpt: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate LaMa on CelebA-HQ 256 test dataset with custom strided masks")
    parser.add_argument('--model-path', type=str, default='lama/lama-fourier-celeba', help='Path to LaMa model directory')
    parser.add_argument('--image-dir', type=str, default='datasets/celeba_hq_256_test', help='Path to test images')
    parser.add_argument('--mask-dir', type=str, default='datasets/testing_mask_dataset', help='Path to testing masks')
    parser.add_argument('--output-dir', type=str, default='evaluation_results_lama', help='Output directory')
    parser.add_argument('--num-images', type=int, default=2000, help='Number of test images to evaluate')
    parser.add_argument('--log-file', type=str, default='lama_evaluation.log', help='Path to save terminal logs')
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
    send_notification(f"LaMa Eval: Started. Total images: {num_images} on device: {device}")
    
    for cat in categories:
        cat_masks = indexed_masks[cat]
        if not cat_masks:
            print(f"No masks found for category {cat}. Skipping.")
            continue
            
        print(f"\nProcessing category: {cat}")
        os.makedirs(os.path.join(args.output_dir, cat), exist_ok=True)
        
        pbar = tqdm(total=num_images)
        for idx in range(num_images):
            img_name = image_files[idx]
            img_path = os.path.join(args.image_dir, img_name)
            
            # Stride 2 mask selection
            mask_idx = (idx * 2) % len(cat_masks)
            mask_path = cat_masks[mask_idx]
            mask_name = os.path.basename(mask_path)
            
            # Load and preprocess image
            img_pil = Image.open(img_path).convert('RGB')
            w, h = img_pil.size
            img_tensor = torch.from_numpy(np.array(img_pil).transpose(2, 0, 1)).float() / 255.0
            
            # Load and preprocess mask
            mask_pil = Image.open(mask_path).convert('L')
            if mask_pil.size != (w, h):
                mask_pil = mask_pil.resize((w, h), Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_pil)).float() / 255.0
            mask_tensor = (mask_tensor > 0.5).float().unsqueeze(0) # 1 x H x W
            
            # Pad image and mask to multiple of 16 for LaMa compatibility
            mod = 16
            pad_h = (mod - h % mod) % mod
            pad_w = (mod - w % mod) % mod
            
            padded_img = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect').unsqueeze(0).to(device)
            padded_mask = torch.nn.functional.pad(mask_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0).unsqueeze(0).to(device)
            
            # Inference
            batch = {
                'image': padded_img,
                'mask': padded_mask
            }
            
            with torch.no_grad():
                batch = model(batch)
                pred_tensor = batch['inpainted'][0]
                
            # Crop back to original resolution
            pred_tensor = pred_tensor[:, :h, :w].cpu()
            
            # Convert to numpy arrays
            pred_np = (pred_tensor.clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            gt_np = np.array(img_pil)
            mask_np = (mask_tensor[0].numpy() * 255.0).astype(np.uint8)
            
            # Compute Merged image
            mask_binary = (mask_tensor[0].numpy() > 0.5).astype(np.uint8)
            mask_binary_3ch = np.expand_dims(mask_binary, axis=-1)
            merged_np = pred_np * mask_binary_3ch + gt_np * (1 - mask_binary_3ch)
            
            # Compute PSNR
            psnr_val = compare_psnr(gt_np, merged_np, data_range=255)
            
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
            pbar.update(1)
            
            # Mobile notification after every 500 images
            if (idx + 1) % 500 == 0:
                send_notification(f"[{cat}] Processed {idx + 1}/{num_images} images | Latest PSNR: {psnr_val:.2f}")
                
        pbar.close()
        send_notification(f"[{cat}] Completed all {num_images} images successfully.")

if __name__ == '__main__':
    main()
