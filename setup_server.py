#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import time
import stat
import json
import urllib.request
import zipfile
import tarfile

def remove_readonly(func, path, excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

# Ensure output is line-buffered for immediate terminal updates
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

TOTAL_STEPS = 13

def draw_progress_bar(step_num, step_name, total_steps):
    columns, _ = shutil.get_terminal_size()
    pct = int((step_num / total_steps) * 100)
    prefix = f"[Step {step_num}/{total_steps}] {step_name}... "
    suffix = f" {pct}%"
    
    bar_width = max(10, columns - len(prefix) - len(suffix) - 5)
    filled_width = int(bar_width * step_num // total_steps)
    
    if filled_width == bar_width:
        bar = "=" * bar_width
    elif filled_width > 0:
        bar = "=" * (filled_width - 1) + ">" + " " * (bar_width - filled_width)
    else:
        bar = " " * bar_width
        
    progress_str = f"{prefix}[{bar}]{suffix}"
    progress_str = progress_str[:columns-1]
    
    sys.stdout.write(f"\r\033[K\033[1;36m{progress_str}\033[0m")
    sys.stdout.flush()

def run_command(cmd_args, step_num, step_name, total_steps, shell=False, cwd=None):
    is_interactive = sys.stdout.isatty()
    
    process = subprocess.Popen(
        cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=shell,
        text=True,
        bufsize=1,
        cwd=cwd,
        env=os.environ
    )
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            line_str = line.rstrip('\r\n')
            if is_interactive:
                # Clear progress line, print log line, redraw progress bar at the bottom
                sys.stdout.write("\r\033[K")
                sys.stdout.write(line_str + "\n")
                draw_progress_bar(step_num, step_name, total_steps)
            else:
                sys.stdout.write(line_str + "\n")
                sys.stdout.flush()
                
    rc = process.poll()
    if rc != 0:
        if is_interactive:
            sys.stdout.write("\n")
        raise subprocess.CalledProcessError(rc, cmd_args)

def get_pip_executable():
    venv_pip = os.path.abspath(os.path.join("venv", "bin", "pip"))
    if os.path.exists(venv_pip):
        return venv_pip
    return "pip"

def execute_step(step_num, step_name, func_or_cmd, shell=False, cwd=None):
    is_interactive = sys.stdout.isatty()
    if is_interactive:
        draw_progress_bar(step_num, step_name, TOTAL_STEPS)
    else:
        print(f"\n=== [Step {step_num}/{TOTAL_STEPS}] {step_name} ===")
        sys.stdout.flush()
        
    try:
        if callable(func_or_cmd):
            func_or_cmd()
            if is_interactive:
                draw_progress_bar(step_num, step_name, TOTAL_STEPS)
        else:
            # If command starts with "pip", use the resolved venv path if available
            cmd = list(func_or_cmd)
            if cmd and cmd[0] == "pip":
                cmd[0] = get_pip_executable()
            run_command(cmd, step_num, step_name, TOTAL_STEPS, shell=shell, cwd=cwd)
    except Exception as e:
        if is_interactive:
            sys.stdout.write("\n")
        print(f"\nERROR in Step {step_num} ({step_name}): {e}", file=sys.stderr)
        sys.exit(1)

# Step definitions
def step_verify_workspace():
    if not os.path.exists("main.py"):
        raise RuntimeError("Could not find main.py. Make sure you are in the correct repository directory.")

def step_setup_cuda():
    cuda_dir = "/usr/local/cuda-12.4"
    if not os.path.exists(cuda_dir):
        raise RuntimeError(f"CUDA 12.4 Toolkit directory not found at {cuda_dir}.")
        
    print(f"Configuring environment to use CUDA Toolkit: {cuda_dir}")
    os.environ["CUDA_HOME"] = cuda_dir
    os.environ["PATH"] = f"{cuda_dir}/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{cuda_dir}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["MAX_JOBS"] = "1"
    os.environ["PIP_NO_CACHE_DIR"] = "1"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    
    # Prepend virtual environment path to PATH if venv already exists
    venv_bin = os.path.abspath("venv/bin")
    if os.path.exists(venv_bin):
        os.environ["PATH"] = f"{venv_bin}:" + os.environ["PATH"]
        
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True, env=os.environ)
        for line in out.splitlines():
            if "release" in line:
                print(f"[OK] CUDA compiler (nvcc) found: {line.strip()}")
                break
    except Exception as e:
        print(f"WARNING: 'nvcc' not found or failed: {e}. Compilation might fail.")

def step_create_venv():
    if os.path.exists("venv"):
        # Check if venv is writable by trying to create a test file inside it
        is_writable = False
        try:
            test_file = os.path.join("venv", ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            is_writable = True
        except Exception:
            pass
            
        if not is_writable:
            print("Virtual environment 'venv' exists but is not writable. Recreating it...")
            try:
                shutil.rmtree("venv", ignore_errors=True)
            except Exception as e:
                print(f"WARNING: Failed to remove 'venv' directory automatically: {e}")
                
            if os.path.exists("venv"):
                raise RuntimeError("Existing 'venv' directory is not writable and could not be deleted. "
                                   "Please clean it up manually by running: sudo rm -rf venv")

    if not os.path.exists("venv"):
        print("Creating virtual environment 'venv'...")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        print("Virtual environment 'venv' created.")
    else:
        print("Virtual environment 'venv' already exists.")
        
    # Append venv/bin to PATH for subsequent steps and subprocesses
    venv_bin = os.path.abspath("venv/bin")
    os.environ["PATH"] = f"{venv_bin}:" + os.environ["PATH"]

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
    
    # Copy across and clean up
    shutil.copytree(os.path.join(temp_dir, "classification", "ops_dcnv3"), target_dir)
    shutil.rmtree(temp_dir, onerror=remove_readonly)
    print("ops_dcnv3 downloaded successfully.")

def check_setuptools_numpy_installed():
    try:
        venv_python = os.path.abspath(os.path.join("venv", "bin", "python"))
        if not os.path.exists(venv_python):
            return False
        out = subprocess.check_output(
            [venv_python, "-c", "import setuptools; import numpy; print(setuptools.__version__, numpy.__version__)"],
            text=True, stderr=subprocess.DEVNULL
        )
        parts = out.strip().split()
        if len(parts) == 2:
            setuptools_ver, numpy_ver = parts
            st_major = int(setuptools_ver.split('.')[0])
            np_major = int(numpy_ver.split('.')[0])
            return st_major < 82 and np_major < 2
    except Exception:
        pass
    return False

def check_pytorch_installed():
    try:
        venv_python = os.path.abspath(os.path.join("venv", "bin", "python"))
        if not os.path.exists(venv_python):
            return False
        out = subprocess.check_output(
            [venv_python, "-c", "import torch; import torchvision; print(torch.__version__, torchvision.__version__)"],
            text=True, stderr=subprocess.DEVNULL
        )
        parts = out.strip().split()
        if len(parts) == 2:
            torch_ver, vision_ver = parts
            return torch_ver.startswith("2.1.2") and vision_ver.startswith("0.16.2")
    except Exception:
        pass
    return False

def check_ninja_packaging_installed():
    try:
        venv_python = os.path.abspath(os.path.join("venv", "bin", "python"))
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
        venv_python = os.path.abspath(os.path.join("venv", "bin", "python"))
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
        venv_python = os.path.abspath(os.path.join("venv", "bin", "python"))
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
    paths = glob.glob(os.path.join("venv", "lib", "python*", "site-packages", "torch", "include", "ATen", "core", "boxing", "impl", "boxing.h"))
    if not paths:
        print("WARNING: Could not find boxing.h to patch. It might not be installed yet, or in a different path.")
        return
    for path in paths:
        try:
            print(f"Checking if {path} needs template parsing patch...")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Normalize newlines
            content_norm = content.replace("\r\n", "\n")
            
            start_marker = "// has_ivalue_to<T> tests the presence/absence of instance method IValue::to<T>()"
            end_marker = "// boxing predicates"
            
            if start_marker in content_norm and end_marker in content_norm:
                start_idx = content_norm.find(start_marker)
                end_idx = content_norm.find(end_marker)
                
                # Check if it is already patched with std::true_type fallback
                if "struct has_ivalue_to : std::true_type {};" in content_norm[start_idx:end_idx]:
                    print(f"[OK] boxing.h is already patched to use std::true_type fallback.")
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
                    print("[OK] Successfully patched boxing.h with true_type fallback.")
            else:
                print(f"WARNING: Markers not found in {path}. Content might be different.")
        except Exception as e:
            print(f"WARNING: Failed to patch {path}: {e}")

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

def step_download_places365_dataset():
    dest_dir = os.path.join("datasets", "places365")
    if os.path.exists(dest_dir) and os.path.isdir(dest_dir) and os.path.exists(os.path.join(dest_dir, "places365_standard", "train")):
        print("Places365 dataset already exists. Skipping download.")
        return
        
    url = "http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar"
    archive_path = os.path.join("datasets", "places365standard_easyformat.tar")
    
    download_file(url, archive_path)
    extract_tar(archive_path, dest_dir)
    
    if os.path.exists(archive_path):
        os.remove(archive_path)

def main():
    is_interactive = sys.stdout.isatty()
    if is_interactive:
        sys.stdout.write("\n")
        sys.stdout.flush()
        
    # Step 1: Verify Workspace
    execute_step(1, "Verifying workspace directory", step_verify_workspace)
    
    # Step 2: Set up CUDA 12.4 Environment
    execute_step(2, "Configuring CUDA 12.4 environment", step_setup_cuda)
    
    # Step 3: Create Venv
    execute_step(3, "Creating Python virtual environment", step_create_venv)
    
    # Step 4: Upgrade pip & wheel
    execute_step(4, "Upgrading pip and wheel", ["pip", "install", "--upgrade", "pip", "wheel"])
    
    # Step 5: Install compatible setuptools & numpy pins
    if check_setuptools_numpy_installed():
        print("[Step 5/13] setuptools < 82 and numpy < 2 already installed. Skipping.")
    else:
        execute_step(5, "Installing setuptools < 82 and numpy < 2", ["pip", "install", "setuptools<82", "numpy<2"])
    
    # Step 6: Install PyTorch 2.1.2 (compatible with CUDA 12.4 compiler)
    if check_pytorch_installed():
        print("[Step 6/13] PyTorch 2.1.2 and torchvision 0.16.2 already installed. Skipping.")
    else:
        execute_step(6, "Installing PyTorch 2.1.2 (CUDA 12.1 whl)", [
            "pip", "install", "torch==2.1.2", "torchvision==0.16.2", 
            "--extra-index-url", "https://download.pytorch.org/whl/cu121"
        ])
    
    # Step 7: Install packaging & ninja
    if check_ninja_packaging_installed():
        print("[Step 7/13] packaging and ninja already installed. Skipping.")
    else:
        execute_step(7, "Installing packaging and ninja compiler tool", ["pip", "install", "packaging", "ninja"])
    
    # Step 8: Compile causal-conv1d & mamba-ssm
    if check_mamba_installed():
        print("[Step 8/13] causal-conv1d and mamba-ssm already compiled and installed. Skipping.")
    else:
        execute_step(8, "Compiling causal-conv1d and mamba-ssm (verbose)", [
            "pip", "install", "causal-conv1d==1.1.3.post1", "mamba-ssm==1.1.3.post1", 
            "--no-build-isolation", "-v"
        ])
    
    # Step 9: Install other requirements
    execute_step(9, "Installing remaining requirements.txt dependencies", ["pip", "install", "-r", "requirements.txt"])
    
    # Step 10: Download InternImage ops_dcnv3
    execute_step(10, "Downloading InternImage ops_dcnv3 folder", step_download_ops)
    
    # Step 11: Compile ops_dcnv3
    if check_dcnv3_compiled():
        print("[Step 11/13] ops_dcnv3 CUDA kernels already compiled and installed. Skipping.")
    else:
        ops_dir = os.path.abspath(os.path.join("src", "ops_dcnv3"))
        make_sh = os.path.join(ops_dir, "make.sh")
        
        try:
            os.chmod(make_sh, 0o755)
        except Exception:
            pass
            
        # Patch PyTorch boxing.h to workaround CUDA 12.4 compile issue
        patch_pytorch_boxing_header()
            
        # Clean build artifacts to ensure a fresh compilation
        build_dir = os.path.join(ops_dir, "build")
        if os.path.exists(build_dir):
            print(f"Cleaning existing build directory: {build_dir}")
            shutil.rmtree(build_dir, ignore_errors=True)
            
        execute_step(11, "Compiling ops_dcnv3 CUDA kernels", ["sh", "make.sh"], cwd=ops_dir)

    # Step 12: Download & extract mask dataset
    execute_step(12, "Download & extract mask dataset", step_download_mask_dataset)
    
    # Step 13: Download & extract Places365 dataset
    execute_step(13, "Download & extract Places365 dataset", step_download_places365_dataset)
    
    if is_interactive:
        sys.stdout.write("\n")
        
    print("\n==========================================================")
    print("          SEM-Net Setup Completed Successfully!           ")
    print("==========================================================")
    print("To activate this virtual environment in the future, run:")
    print("  source venv/bin/activate")
    print("\nTo run training, update config.yml with local datasets, then:")
    print("  python train.py")
    print("==========================================================\n")

if __name__ == "__main__":
    main()
