#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import stat
import time

def remove_readonly(func, path, excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

# Ensure output is line-buffering for immediate terminal updates
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

TOTAL_STEPS = 12

def execute_step(step_num, step_name, func_or_cmd, shell=False, cwd=None):
    print(f"\n=== [Step {step_num}/{TOTAL_STEPS}] {step_name} ===")
    sys.stdout.flush()

    try:
        if callable(func_or_cmd):
            func_or_cmd()
        else:
            cmd = list(func_or_cmd)
            if cmd and cmd[0] == "pip":
                venv_pip = os.path.abspath(os.path.join("venv", "bin", "pip"))
                if sys.platform == "win32":
                    venv_pip = os.path.abspath(os.path.join("venv", "Scripts", "pip.exe"))
                if os.path.exists(venv_pip):
                    cmd[0] = venv_pip
            
            subprocess.check_call(cmd, shell=shell, cwd=cwd, env=os.environ)

    except Exception as e:
        print(f"\nERROR in Step {step_num} ({step_name}): {e}", file=sys.stderr)
        raise

# Step definitions
def step_verify_workspace():
    if not os.path.exists("main.py"):
        raise RuntimeError("Could not find main.py. Make sure you are in the correct directory (LAST).")

def step_setup_cuda():
    cuda_dir = "/usr/local/cuda-12.4"
    if sys.platform != "win32" and not os.path.exists(cuda_dir):
        print(f"WARNING: CUDA 12.4 Toolkit directory not found at {cuda_dir}. Skipping CUDA configuration.")
        return
    if sys.platform != "win32":
        print(f"Configuring environment to use CUDA Toolkit: {cuda_dir}")
        os.environ["CUDA_HOME"] = cuda_dir
        os.environ["PATH"] = f"{cuda_dir}/bin:" + os.environ.get("PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_dir}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    
    os.environ["MAX_JOBS"] = "1"
    os.environ["PIP_NO_CACHE_DIR"] = "1"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    
    # Prepend virtual environment path to PATH if venv already exists
    if sys.platform == "win32":
        venv_bin = os.path.abspath(os.path.join("venv", "Scripts"))
        path_sep = ";"
    else:
        venv_bin = os.path.abspath(os.path.join("venv", "bin"))
        path_sep = ":"
        
    if os.path.exists(venv_bin):
        os.environ["PATH"] = f"{venv_bin}{path_sep}" + os.environ.get("PATH", "")

def step_create_venv():
    if not os.path.exists("venv"):
        print("Creating virtual environment 'venv'...")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        print("Virtual environment 'venv' created.")
    else:
        print("Virtual environment 'venv' already exists.")
        
    if sys.platform == "win32":
        venv_bin = os.path.abspath(os.path.join("venv", "Scripts"))
        path_sep = ";"
    else:
        venv_bin = os.path.abspath(os.path.join("venv", "bin"))
        path_sep = ":"
    os.environ["PATH"] = f"{venv_bin}{path_sep}" + os.environ.get("PATH", "")

def step_download_ops():
    target_dir = os.path.join("src", "ops_dcnv3")
    if os.path.exists(target_dir) and os.path.exists(os.path.join(target_dir, "make.sh")):
        print("src/ops_dcnv3 already exists. Skipping download.")
        return
        
    print("Downloading ops_dcnv3 from InternImage repository...")
    if os.path.lexists(target_dir):
        if os.path.isdir(target_dir) and not os.path.islink(target_dir):
            shutil.rmtree(target_dir, onerror=remove_readonly)
        else:
            os.remove(target_dir)
            
    temp_dir = "temp_internimage"
    if os.path.lexists(temp_dir):
        if os.path.isdir(temp_dir) and not os.path.islink(temp_dir):
            shutil.rmtree(temp_dir, onerror=remove_readonly)
        else:
            os.remove(temp_dir)
    os.makedirs(temp_dir)
    
    subprocess.check_call(["git", "init", "-q"], cwd=temp_dir)
    subprocess.check_call(["git", "remote", "add", "origin", "https://github.com/OpenGVLab/InternImage.git"], cwd=temp_dir)
    subprocess.check_call(["git", "config", "core.sparseCheckout", "true"], cwd=temp_dir)
    
    sparse_path = os.path.join(temp_dir, ".git", "info", "sparse-checkout")
    with open(sparse_path, "w") as f:
        f.write("classification/ops_dcnv3\n")
        
    subprocess.check_call(["git", "pull", "-q", "--depth", "1", "origin", "master"], cwd=temp_dir)
    
    shutil.copytree(os.path.join(temp_dir, "classification", "ops_dcnv3"), target_dir)
    shutil.rmtree(temp_dir, onerror=remove_readonly)
    print("ops_dcnv3 downloaded successfully.")

def get_venv_python():
    if sys.platform == "win32":
        return os.path.abspath(os.path.join("venv", "Scripts", "python.exe"))
    return os.path.abspath(os.path.join("venv", "bin", "python"))

def check_setuptools_numpy_installed():
    try:
        venv_python = get_venv_python()
        if not os.path.exists(venv_python):
            return False
        out = subprocess.check_output(
            [venv_python, "-c", "import setuptools; import numpy; print(setuptools.__version__, numpy.__version__)"],
            text=True, stderr=subprocess.DEVNULL
        )
        parts = out.strip().split()
        if len(parts) == 2:
            st_major = int(parts[0].split('.')[0])
            np_major = int(parts[1].split('.')[0])
            return st_major < 82 and np_major < 2
    except Exception:
        pass
    return False

def check_pytorch_installed():
    try:
        venv_python = get_venv_python()
        if not os.path.exists(venv_python):
            return False
        out = subprocess.check_output(
            [venv_python, "-c", "import torch; import torchvision; print(torch.__version__, torchvision.__version__)"],
            text=True, stderr=subprocess.DEVNULL
        )
        parts = out.strip().split()
        if len(parts) == 2:
            return parts[0].startswith("2.1.2") and parts[1].startswith("0.16.2")
    except Exception:
        pass
    return False

def check_ninja_packaging_installed():
    try:
        venv_python = get_venv_python()
        if not os.path.exists(venv_python):
            return False
        subprocess.check_call(
            [venv_python, "-c", "import packaging; import ninja"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        pass
    return False

def check_mamba_installed():
    try:
        venv_python = get_venv_python()
        if not os.path.exists(venv_python):
            return False
        subprocess.check_call(
            [venv_python, "-c", "import causal_conv1d.causal_conv1d_interface; import mamba_ssm"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        pass
    return False

def check_dcnv3_compiled():
    try:
        venv_python = get_venv_python()
        if not os.path.exists(venv_python):
            return False
        subprocess.check_call(
            [venv_python, "-c", "import torch; import DCNv3"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        pass
    return False

def patch_pytorch_boxing_header():
    import glob
    site_packages = os.path.join("venv", "Lib", "site-packages") if sys.platform == "win32" else os.path.join("venv", "lib", "python*", "site-packages")
    paths = glob.glob(os.path.join(site_packages, "torch", "include", "ATen", "core", "boxing", "impl", "boxing.h"))
    if not paths:
        print("WARNING: Could not find boxing.h to patch.")
        return
    for path in paths:
        try:
            print(f"Checking if {path} needs template parsing patch...")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            content_norm = content.replace("\r\n", "\n")
            start_marker = "// has_ivalue_to<T> tests the presence/absence of instance method IValue::to<T>()"
            end_marker = "// boxing predicates"
            
            if start_marker in content_norm and end_marker in content_norm:
                start_idx = content_norm.find(start_marker)
                end_idx = content_norm.find(end_marker)
                
                if "struct has_ivalue_to : std::true_type {};" in content_norm[start_idx:end_idx]:
                    print(f"[OK] boxing.h is already patched.")
                else:
                    print(f"Patching boxing.h to use std::true_type fallback in: {path}")
                    replacement_block = (
                        "// has_ivalue_to<T> tests the presence/absence of instance method IValue::to<T>()\n"
                        "//\n"
                        "template <class T, class Enable = void>\n"
                        "struct has_ivalue_to : std::true_type {};\n\n"
                    )
                    new_content = content_norm[:start_idx] + replacement_block + content_norm[end_idx:]
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_content)
            else:
                print(f"WARNING: Markers not found in {path}. Content might be different.")
        except Exception as e:
            print(f"WARNING: Failed to patch {path}: {e}")

import urllib.request
import zipfile
import tarfile

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
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
            if current_time - last_update_time > 2.0 or downloaded == total_size:
                last_update_time = current_time
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    msg = f"Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)"
                else:
                    msg = f"Downloaded {downloaded / (1024*1024):.1f} MB"
                print(msg)
                sys.stdout.flush()

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

def step_download_mask_dataset():
    dest_dir = os.path.join("datasets", "testing_mask_dataset")
    if os.path.exists(dest_dir) and os.path.isdir(dest_dir) and any(os.path.isfile(os.path.join(dest_dir, f)) for f in os.listdir(dest_dir) if not f.startswith('.')):
        print("Mask dataset already exists. Skipping download.")
        return
        
    url = "https://files.lalithadithyan.dev/download/testing_mask_dataset.zip"
    archive_path = os.path.join("datasets", "testing_mask_dataset.zip")
    
    download_file(url, archive_path)
    extract_zip(archive_path, "datasets")
    
    if os.path.exists(archive_path):
        os.remove(archive_path)

def main():
    try:
        execute_step(1, "Verifying workspace directory", step_verify_workspace)
        execute_step(2, "Configuring CUDA 12.4 environment", step_setup_cuda)
        execute_step(3, "Creating Python virtual environment", step_create_venv)
        execute_step(4, "Upgrading pip and wheel", ["pip", "install", "--upgrade", "pip", "wheel"])
        
        if check_setuptools_numpy_installed():
            print("[Step 5/10] setuptools < 82 and numpy < 2 already installed. Skipping.")
        else:
            execute_step(5, "Installing setuptools < 82 and numpy < 2", ["pip", "install", "setuptools<82", "numpy<2"])
        
        if check_pytorch_installed():
            print("[Step 6/10] PyTorch 2.1.2 and torchvision 0.16.2 already installed. Skipping.")
        else:
            execute_step(6, "Installing PyTorch 2.1.2 (CUDA 12.1 whl)", [
                "pip", "install", "torch==2.1.2", "torchvision==0.16.2",
                "--extra-index-url", "https://download.pytorch.org/whl/cu121"
            ])

        if check_ninja_packaging_installed():
            print("[Step 7/10] packaging and ninja already installed. Skipping.")
        else:
            execute_step(7, "Installing packaging and ninja compiler tool", ["pip", "install", "packaging", "ninja"])

        if check_mamba_installed():
            print("[Step 8/10] causal-conv1d and mamba-ssm already compiled. Skipping.")
        else:
            execute_step(8, "Compiling causal-conv1d and mamba-ssm", [
                "pip", "install", "causal-conv1d==1.1.3.post1", "mamba-ssm==1.1.3.post1",
                "--no-build-isolation", "-v"
            ])

        execute_step(9, "Installing remaining requirements.txt dependencies", ["pip", "install", "-r", "requirements.txt"])
        execute_step(10, "Downloading InternImage ops_dcnv3 folder", step_download_ops)

        if check_dcnv3_compiled():
            print("ops_dcnv3 CUDA kernels already compiled. Skipping.")
        else:
            ops_dir = os.path.abspath(os.path.join("src", "ops_dcnv3"))
            if sys.platform != "win32":
                make_sh = os.path.join(ops_dir, "make.sh")
                try: os.chmod(make_sh, 0o755)
                except: pass
            patch_pytorch_boxing_header()
            build_dir = os.path.join(ops_dir, "build")
            if os.path.exists(build_dir):
                shutil.rmtree(build_dir, ignore_errors=True)
            execute_step(11, "Compiling ops_dcnv3 CUDA kernels", [get_venv_python(), "setup.py", "build", "install"], cwd=ops_dir)

        execute_step(12, "Download & extract mask dataset", step_download_mask_dataset)

    except BaseException as e:
        sys.exit(1)

    print("\n==========================================================")
    print("          Setup Completed Successfully!           ")
    print("==========================================================")
    print("Launching training script: train_local.sh / train_local.bat")
    print("==========================================================")

    if sys.platform == "win32":
        training_script = os.path.abspath("train_local.bat")
        os.system(f'"{training_script}"')
    else:
        training_script = os.path.abspath("train_local.sh")
        try: os.chmod(training_script, 0o755)
        except: pass
        os.execv("/bin/bash", ["/bin/bash", training_script])

if __name__ == "__main__":
    main()
