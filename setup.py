#!/usr/bin/env python3
import os
try:
    import getpass
    username = getpass.getuser()
except Exception:
    username = "semnet_user"
os.environ["TORCH_HOME"] = f"/tmp/{username}/torch_cache"
os.environ["MPLCONFIGDIR"] = f"/tmp/{username}/matplotlib_cache"

import sys
import subprocess
import shutil
import time
import json
import urllib.request
import zipfile
import tarfile
import re
import glob

# Ensure output is printed immediately
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
        print(f"[OK] Package '{package_name}' is installed and importable.")
        return True
    except ImportError:
        print(f"[WARNING] Package '{package_name}' is NOT installed.")
        return False

def verify_all_dependencies():
    print("\n--- Verifying Dependencies ---")
    required = {
        "torch": "torch",
        "torchvision": "torchvision",
        "causal-conv1d": "causal_conv1d",
        "mamba-ssm": "mamba_ssm",
        "DCNv3": "DCNv3",
        "scipy": "scipy",
        "imageio": "imageio",
        "opencv-python": "cv2",
        "scikit-image": "skimage",
        "lpips": "lpips",
        "cleanfid": "cleanfid",
        "matplotlib": "matplotlib",
        "requests": "requests"
    }
    
    all_ok = True
    missing_packages = []
    for pkg, imp in required.items():
        if not check_package(pkg, imp):
            all_ok = False
            missing_packages.append(pkg)
            
    # Check CUDA support in PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] PyTorch CUDA support is available. Device count: {torch.cuda.device_count()}")
        else:
            print("[WARNING] PyTorch CUDA support is NOT available. Running on CPU.")
    except Exception:
        pass
        
    if not all_ok:
        print(f"\n[WARNING] Missing packages: {', '.join(missing_packages)}")
        # Check if running in venv, if so we can try installing
        if hasattr(sys, 'real_prefix') or (getattr(sys, 'base_prefix', sys.prefix) != sys.prefix):
            print("Running inside a virtual environment. Installing missing requirements...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                print("[OK] Requirements installed successfully.")
                all_ok = True
            except Exception as e:
                print(f"[ERROR] Failed to install requirements automatically: {e}")
        else:
            print("Please activate your virtual environment or install the requirements manually:")
            print("  pip install -r requirements.txt")
            
    return all_ok

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, 'wb') as out_file:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1MB
            
            last_update_time = time.time()
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                
                current_time = time.time()
                if current_time - last_update_time > 5.0 or downloaded == total_size:
                    last_update_time = current_time
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        msg = f"Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)"
                    else:
                        msg = f"Downloaded {downloaded / (1024*1024):.1f} MB"
                    print(msg)
                    sys.stdout.flush()
        return True
    except Exception as e:
        print(f"Error downloading from {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction completed successfully.")

def extract_tar(tar_path, extract_to):
    print(f"Extracting {tar_path} to {extract_to} (this may take several minutes)...")
    os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(tar_path, 'r') as tar_ref:
        tar_ref.extractall(extract_to)
    print("Extraction completed successfully.")

def verify_datasets():
    print("\n--- Verifying Datasets ---")
    
    # 1. Mask Dataset
    mask_dir = os.path.join("datasets", "testing_mask_dataset")
    if os.path.exists(mask_dir) and os.path.isdir(mask_dir) and any(os.path.isfile(os.path.join(mask_dir, f)) for f in os.listdir(mask_dir) if not f.startswith('.')):
        print("[OK] Mask dataset exists.")
    else:
        print("Mask dataset missing. Attempting download...")
        url = "https://files.lalithadithyan.dev/download/testing_mask_dataset.zip"
        archive_path = os.path.join("datasets", "testing_mask_dataset.zip")
        if download_file(url, archive_path):
            extract_zip(archive_path, "datasets")
            if os.path.exists(archive_path):
                os.remove(archive_path)
        else:
            print("[ERROR] Failed to download mask dataset.")
            
    # 2. Places365 Test Dataset
    places_dir = os.path.join("datasets", "places365")
    test_dir = os.path.join(places_dir, "test_256")
    if os.path.exists(test_dir) and os.path.isdir(test_dir) and len(os.listdir(test_dir)) > 0:
        print("[OK] Places365 test dataset exists.")
    else:
        print("Places365 test dataset missing. Attempting download...")
        archive_path = os.path.join("datasets", "test_256.tar")
        
        # Check local tar first
        local_tar_root = "test_256.tar"
        if os.path.exists(local_tar_root):
            print(f"Found local test_256.tar in root. Moving it to {archive_path}...")
            os.makedirs("datasets", exist_ok=True)
            try:
                shutil.move(local_tar_root, archive_path)
            except Exception as e:
                print(f"WARNING: Failed to move local tar: {e}")
                
        if not os.path.exists(archive_path):
            # Try Google VPS files server first
            g_url = "https://files.lalithadithyan.dev/download/test_256.tar"
            print("Trying download from files server...")
            success = download_file(g_url, archive_path)
            if not success:
                # Fallback to MIT CSAIL server
                m_url = "http://data.csail.mit.edu/places/places365/test_256.tar"
                print("Fallback: downloading from MIT CSAIL server...")
                success = download_file(m_url, archive_path)
                
            if success:
                extract_tar(archive_path, places_dir)
                if os.path.exists(archive_path):
                    os.remove(archive_path)
            else:
                print("[ERROR] Failed to download Places365 test dataset.")

def verify_checkpoints():
    print("\n--- Verifying Checkpoints ---")
    run_path = "./PlacesTraining"
    os.makedirs(run_path, exist_ok=True)
    
    gen_dest = os.path.join(run_path, "InpaintingModel_gen.pth")
    if os.path.exists(gen_dest):
        print(f"[OK] InpaintingModel generator checkpoint exists at {gen_dest}.")
        return True
        
    print("Checkpoint missing. Attempting to restore...")
    # Check local root checkpoints first
    local_gens = glob.glob("*_InpaintingModel_gen.pth")
    local_dics = glob.glob("*_InpaintingModel_dis.pth")
    
    if local_gens:
        gen_files_parsed = []
        for f in local_gens:
            m = re.match(r'^(\d+)_InpaintingModel_gen\.pth$', f)
            if m:
                gen_files_parsed.append((int(m.group(1)), f))
        if gen_files_parsed:
            best_iter, best_gen_name = max(gen_files_parsed, key=lambda x: x[0])
            print(f"Found local generator checkpoint in root: {best_gen_name}")
            shutil.copy(best_gen_name, gen_dest)
            
            dis_dest = os.path.join(run_path, "InpaintingModel_dis.pth")
            dis_name = f"{best_iter:09d}_InpaintingModel_dis.pth"
            if os.path.exists(dis_name):
                shutil.copy(dis_name, dis_dest)
            elif local_dics:
                dis_files_parsed = []
                for f in local_dics:
                    m = re.match(r'^(\d+)_InpaintingModel_dis\.pth$', f)
                    if m:
                        dis_files_parsed.append((int(m.group(1)), f))
                if dis_files_parsed:
                    _, best_dis_name = max(dis_files_parsed, key=lambda x: x[0])
                    shutil.copy(best_dis_name, dis_dest)
            print("[OK] Local checkpoint restored successfully.")
            return True
            
    # Try files server download
    files_url = os.environ.get("FILES_SERVER_URL", "https://files.lalithadithyan.dev")
    session = os.environ.get("C2_SESSION", "Places")
    list_url = f"{files_url}/api/models?session={session}"
    
    print(f"Fetching model listing from: {list_url}")
    try:
        req = urllib.request.Request(list_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            
        file_list = data.get("files", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        
        if file_list:
            gen_files = [(int(m.group(1)), f) for f in file_list
                         for m in [re.match(r'^(\d{9})_InpaintingModel_gen\.pth$', f)] if m]
            dis_files = [(int(m.group(1)), f) for f in file_list
                         for m in [re.match(r'^(\d{9})_InpaintingModel_dis\.pth$', f)] if m]
                         
            if gen_files and dis_files:
                best_iter_gen, best_gen_name = max(gen_files, key=lambda x: x[0])
                best_iter_dis, best_dis_name = max(dis_files, key=lambda x: x[0])
                
                if best_iter_gen != best_iter_dis:
                    common_iter = min(best_iter_gen, best_iter_dis)
                    candidates_gen = [x for x in gen_files if x[0] == common_iter]
                    candidates_dis = [x for x in dis_files if x[0] == common_iter]
                    if candidates_gen and candidates_dis:
                        best_gen_name = candidates_gen[0][1]
                        best_dis_name = candidates_dis[0][1]
                        best_iter_gen = common_iter
                        
                print(f"Downloading checkpoint iteration {best_iter_gen:,}...")
                download_file(f"{files_url}/download/{best_gen_name}", gen_dest)
                
                dis_dest = os.path.join(run_path, "InpaintingModel_dis.pth")
                download_file(f"{files_url}/download/{best_dis_name}", dis_dest)
                print("[OK] Checkpoint downloaded successfully.")
                return True
    except Exception as e:
        print(f"Failed to fetch/download checkpoint from server: {e}")
        
    print("[ERROR] No checkpoint files available.")
    return False

def main():
    print("==========================================")
    print("      SEM-Net System Setup & Verification ")
    print("==========================================")
    
    # 1. Verify environment
    verify_all_dependencies()
    
    # 2. Verify datasets
    verify_datasets()
    
    # 3. Verify checkpoints
    verify_checkpoints()
    
    # 4. Start testing
    print("\n==========================================")
    print("           Starting Evaluation            ")
    print("==========================================")
    
    # Resolve the python executable inside the virtual environment if present
    python_bin = sys.executable
    if os.path.exists("venv"):
        venv_bin_python = os.path.abspath(os.path.join("venv", "bin", "python"))
        venv_win_python = os.path.abspath(os.path.join("venv", "Scripts", "python.exe"))
        if os.path.exists(venv_bin_python):
            python_bin = venv_bin_python
        elif os.path.exists(venv_win_python):
            python_bin = venv_win_python

    eval_cmd = [
        python_bin,
        "evaluate_test.py",
        "--input-size", "256",
        "--batch-size", "128",
        "--fast-metrics-only"
    ]
    
    # Add any extra arguments passed to setup.py
    if len(sys.argv) > 1:
        eval_cmd.extend(sys.argv[1:])
        
    print(f"Running command: {' '.join(eval_cmd)}")
    sys.stdout.flush()
    
    try:
        subprocess.check_call(eval_cmd)
        print("\n[OK] Evaluation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Evaluation failed with exit code: {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
