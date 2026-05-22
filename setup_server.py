#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import time

# Ensure output is line-buffered for immediate terminal updates
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

TOTAL_STEPS = 11

def draw_progress_bar(step_num, step_name, total_steps):
    columns, _ = shutil.get_terminal_size()
    pct = int((step_num / total_steps) * 100)
    prefix = f"[Step {step_num}/{total_steps}] {step_name}... "
    suffix = f" {pct}%"
    
    # 5 is for bracket styling: " [] "
    bar_width = max(10, columns - len(prefix) - len(suffix) - 5)
    filled_width = int(bar_width * step_num // total_steps)
    
    if filled_width == bar_width:
        bar = "=" * bar_width
    elif filled_width > 0:
        bar = "=" * (filled_width - 1) + ">" + " " * (bar_width - filled_width)
    else:
        bar = " " * bar_width
        
    progress_str = f"{prefix}[{bar}]{suffix}"
    # Truncate to avoid line wrapping issues
    progress_str = progress_str[:columns-1]
    
    # Draw cyan progress bar
    sys.stdout.write(f"\r\033[K\033[1;36m{progress_str}\033[0m")
    sys.stdout.flush()

def run_command(cmd_args, step_num, step_name, total_steps, shell=False, cwd=None):
    is_interactive = sys.stdout.isatty()
    
    # Ensure current environment variables are passed to subprocess
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
    # Resolve absolute path to virtual environment pip if it exists
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
        print("src/ops_dcnv3 already exists (tracked in repository). Skipping download.")
        return
        
    print("Downloading ops_dcnv3 from InternImage repository...")
    if os.path.lexists(target_dir):
        if os.path.isdir(target_dir) and not os.path.islink(target_dir):
            shutil.rmtree(target_dir)
        else:
            os.remove(target_dir)
            
    temp_dir = "temp_internimage"
    if os.path.lexists(temp_dir):
        if os.path.isdir(temp_dir) and not os.path.islink(temp_dir):
            shutil.rmtree(temp_dir)
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
    shutil.rmtree(temp_dir)
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
            [venv_python, "-c", "import causal_conv1d; import mamba_ssm"],
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
        print("[Step 5/11] setuptools < 82 and numpy < 2 already installed. Skipping.")
    else:
        execute_step(5, "Installing setuptools < 82 and numpy < 2", ["pip", "install", "setuptools<82", "numpy<2"])
    
    # Step 6: Install PyTorch 2.1.2 (compatible with CUDA 12.4 compiler)
    if check_pytorch_installed():
        print("[Step 6/11] PyTorch 2.1.2 and torchvision 0.16.2 already installed. Skipping.")
    else:
        execute_step(6, "Installing PyTorch 2.1.2 (CUDA 12.1 whl)", [
            "pip", "install", "torch==2.1.2", "torchvision==0.16.2", 
            "--extra-index-url", "https://download.pytorch.org/whl/cu121"
        ])
    
    # Step 7: Install packaging & ninja
    if check_ninja_packaging_installed():
        print("[Step 7/11] packaging and ninja already installed. Skipping.")
    else:
        execute_step(7, "Installing packaging and ninja compiler tool", ["pip", "install", "packaging", "ninja"])
    
    # Step 8: Compile causal-conv1d & mamba-ssm
    if check_mamba_installed():
        print("[Step 8/11] causal-conv1d and mamba-ssm already compiled and installed. Skipping.")
    else:
        execute_step(8, "Compiling causal-conv1d and mamba-ssm (verbose)", [
            "pip", "install", "causal-conv1d>=1.1.0", "mamba-ssm==1.1.3.post1", 
            "--no-build-isolation", "-v"
        ])
    
    # Step 9: Install other requirements
    execute_step(9, "Installing remaining requirements.txt dependencies", ["pip", "install", "-r", "requirements.txt"])
    
    # Step 10: Download InternImage ops_dcnv3
    execute_step(10, "Downloading InternImage ops_dcnv3 folder", step_download_ops)
    
    # Step 11: Compile ops_dcnv3
    if check_dcnv3_compiled():
        print("[Step 11/11] ops_dcnv3 CUDA kernels already compiled and installed. Skipping.")
    else:
        ops_dir = os.path.abspath(os.path.join("src", "ops_dcnv3"))
        make_sh = os.path.join(ops_dir, "make.sh")
        
        try:
            os.chmod(make_sh, 0o755)
        except Exception:
            pass
            
        execute_step(11, "Compiling ops_dcnv3 CUDA kernels", ["sh", "make.sh"], cwd=ops_dir)
    
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
