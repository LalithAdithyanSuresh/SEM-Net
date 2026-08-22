#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import time
import stat
import json
import socket
import threading
import urllib.request
import zipfile
import tarfile
import ssl
import queue
import re

_printed_warning = False

def remove_readonly(func, path, excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

# Ensure output is line-buffered for immediate terminal updates
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

TOTAL_STEPS = 12

# ── Live Status Reporter ───────────────────────────────────────────────────
# Posts live step progress to the C2 server for the dashboard.
_reporter = None  # initialized in main()

class StatusReporter:
    def __init__(self, c2_url, session):
        self.c2_url     = c2_url.rstrip('/')
        self.session    = session
        self.started_at = time.time()
        self.host       = socket.gethostname()
        self._lock      = threading.Lock()
        self.steps      = {
            i: {'num': i, 'name': self.STEP_NAMES_DEFAULT[i-1], 'status': 'pending',
                'started_at': None, 'finished_at': None, 'duration': None, 'logs': []}
            for i in range(1, TOTAL_STEPS + 1)
        }
        self._queue     = queue.Queue()
        # Start a single worker thread to serialize updates and enforce delays
        threading.Thread(target=self._worker, daemon=True).start()
        self._post(self._payload())  # register immediately

    STEP_NAMES_DEFAULT = [
        'Verifying workspace directory',
        'Configuring CUDA environment',
        'Creating Python virtual environment',
        'Upgrading pip and wheel',
        'Installing setuptools < 82 and numpy < 2',
        'Installing PyTorch 2.1.2',
        'Installing packaging and ninja',
        'Compiling causal-conv1d and mamba-ssm',
        'Installing remaining requirements.txt',
        'Downloading InternImage ops_dcnv3',
        'Compiling ops_dcnv3 CUDA kernels',
        'Setting up Places2 dataset & PlacesTraining config',
    ]

    def _read_history(self):
        try:
            if not os.path.exists("setup_history.txt"):
                return []
            with open("setup_history.txt", "r") as f:
                return f.read().splitlines()
        except Exception:
            return []

    def _payload(self):
        with self._lock:
            hist = self._read_history()
            recent_logs = hist[-10:] if len(hist) > 10 else hist
            return {
                'session': self.session,
                'host': self.host,
                'started_at': self.started_at,
                'updated_at': time.time(),
                'steps': list(self.steps.values()),
                'terminal_logs': recent_logs
            }

    def _post(self, data):
        self._queue.put(data)

    def _worker(self):
        while True:
            data = self._queue.get()
            try:
                url = f"{self.c2_url}/api/setup/status"
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode('utf-8'),
                    headers={'Content-Type': 'application/json', 'User-Agent': 'StatusReporter/1.0'}
                )
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                with urllib.request.urlopen(req, context=ctx, timeout=5) as response:
                    response.read()
            except Exception as e:
                global _printed_warning
                if not _printed_warning:
                    sys.stderr.write(f"[Dashboard Warning] Failed to post status to {self.c2_url}: {e}\n")
                    _printed_warning = True
            finally:
                self._queue.task_done()
                time.sleep(0.1)

    def start_step(self, num):
        with self._lock:
            if num in self.steps:
                self.steps[num]['status'] = 'running'
                self.steps[num]['started_at'] = time.time()
        self._post(self._payload())

    def complete_step(self, num):
        with self._lock:
            if num in self.steps:
                step = self.steps[num]
                step['status'] = 'completed'
                step['finished_at'] = time.time()
                if step['started_at']:
                    step['duration'] = step['finished_at'] - step['started_at']
        self._post(self._payload())

    def skip_step(self, num, reason=None):
        with self._lock:
            if num in self.steps:
                step = self.steps[num]
                step['status'] = 'skipped'
                if reason:
                    step['logs'].append(f"Skipped: {reason}")
        self._post(self._payload())

    def fail_step(self, num, err_msg):
        with self._lock:
            if num in self.steps:
                step = self.steps[num]
                step['status'] = 'failed'
                step['finished_at'] = time.time()
                if step['started_at']:
                    step['duration'] = step['finished_at'] - step['started_at']
                step['logs'].append(f"Error: {err_msg}")
        self._post(self._payload())


# ── Exec Step Wrapper ───────────────────────────────────────────────────────
def execute_step(step_num, step_name, func_or_cmd, cwd=None, shell=False):
    def get_pip_executable():
        venv_pip = os.path.abspath(os.path.join("venv", "bin", "pip"))
        if os.path.exists(venv_pip):
            return venv_pip
        return "pip"

    def run_command(cmd, cwd=None, shell=False):
        print(f"Executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        sys.stdout.flush()
        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            shell=shell,
            env=os.environ
        )
        
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if _reporter:
                with _reporter._lock:
                    _reporter.steps[step_num]['logs'].append(line.rstrip('\n'))
        
        p.wait()
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd)

    is_interactive = sys.stdout.isatty()
    header_step = f"[Step {step_num}/{TOTAL_STEPS}] {step_name}..."
    if is_interactive:
        progress = int((step_num - 1) / TOTAL_STEPS * 50)
        bar = '=' * progress + '>' + ' ' * (49 - progress)
        sys.stdout.write(f"\r{header_step:<60} [{bar}] {int((step_num-1)/TOTAL_STEPS*100)}%")
        sys.stdout.flush()
    else:
        print(f"\n=== {header_step} ===")
        sys.stdout.flush()

    if _reporter:
        _reporter.start_step(step_num)

    try:
        if callable(func_or_cmd):
            func_or_cmd()
        else:
            cmd = list(func_or_cmd)
            if cmd and cmd[0] == "pip":
                cmd[0] = get_pip_executable()
            run_command(cmd, cwd=cwd, shell=shell)

        if _reporter:
            _reporter.complete_step(step_num)

    except Exception as e:
        if _reporter:
            _reporter.fail_step(step_num, str(e))
        if is_interactive:
            sys.stdout.write("\n")
        print(f"\nERROR in Step {step_num} ({step_name}): {e}", file=sys.stderr)
        raise


# Step definitions
def step_verify_workspace():
    if not os.path.exists("main.py"):
        raise RuntimeError("Could not find main.py. Make sure you are in the correct repository directory.")

def step_setup_cuda():
    cuda_dir = "/usr/local/cuda-12.4"
    if not os.path.exists(cuda_dir):
        if os.path.exists("/usr/local/cuda"):
            cuda_dir = "/usr/local/cuda"
        else:
            nvcc_path = shutil.which("nvcc")
            if nvcc_path:
                cuda_dir = os.path.dirname(os.path.dirname(nvcc_path))
            else:
                import glob
                cuda_paths = glob.glob("/usr/local/cuda-*")
                if cuda_paths:
                    cuda_paths.sort()
                    cuda_dir = cuda_paths[-1]
                else:
                    raise RuntimeError("CUDA Toolkit directory not found. Please set CUDA_HOME environment variable manually.")
                    
    print(f"Configuring environment to use CUDA Toolkit: {cuda_dir}")
    os.environ["CUDA_HOME"] = cuda_dir
    os.environ["PATH"] = f"{cuda_dir}/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{cuda_dir}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["MAX_JOBS"] = "1"
    os.environ["PIP_NO_CACHE_DIR"] = "1"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    os.environ["PIP_CACHE_DIR"] = os.path.abspath("tmp/pip_cache")
    os.environ["TORCH_HOME"] = os.path.abspath("tmp/torch_cache")
    os.environ["MPLCONFIGDIR"] = os.path.abspath("tmp/matplotlib_cache")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
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
        raise RuntimeError(f"'nvcc' not found or failed: {e}. CUDA compilation support is required.")

def step_create_venv():
    if os.path.exists("venv"):
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
                raise RuntimeError("Existing 'venv' directory is not writable and could not be deleted.")

    if not os.path.exists("venv"):
        print("Creating virtual environment 'venv'...")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        print("Virtual environment 'venv' created.")
    else:
        print("Virtual environment 'venv' already exists.")
        
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
        raise RuntimeError("Could not find boxing.h header file in PyTorch site-packages for ops_dcnv3 patching.")
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

def step_setup_places_dataset_and_config():
    import yaml
    os.makedirs("datasets", exist_ok=True)
    places_dir = os.path.join("datasets", "places365")
    
    # 1. Locate or symlink Places365
    if not os.path.exists(places_dir):
        server_dirs = [
            "/tmp/cks/CAMINO-DAVA/SEM-Net/datasets/places365",
            "/mnt/datadrive/inpaint/places2/places365standard_easyformat"
        ]
        found = None
        for d in server_dirs:
            if os.path.exists(d):
                found = d
                break
        if found:
            print(f"Found Places365 dataset at {found}. Creating symlink...")
            try:
                os.symlink(found, places_dir)
            except Exception:
                shutil.copytree(found, places_dir)
        else:
            print("Places365 dataset not found locally. Downloading tar archive...")
            tar_path = os.path.join("datasets", "places365standard_easyformat.tar")
            files_url = os.environ.get('FILES_SERVER_URL', 'https://files.lalithadithyan.dev')
            urllib.request.urlretrieve(f"{files_url}/download/places365standard_easyformat.tar", tar_path)
            print("Extracting Places365 dataset...")
            with tarfile.open(tar_path) as tar:
                tar.extractall(path="datasets")
            if os.path.exists("datasets/places365standard_easyformat") and not os.path.exists(places_dir):
                os.rename("datasets/places365standard_easyformat", places_dir)
            if os.path.exists(tar_path):
                os.remove(tar_path)
    else:
        print("Places365 dataset already present.")

    # 2. Testing Mask Dataset
    mask_dir = os.path.join("datasets", "testing_mask_dataset")
    if not os.path.exists(mask_dir):
        print("Downloading testing mask dataset...")
        zip_path = os.path.join("datasets", "testing_mask_dataset.zip")
        files_url = os.environ.get('FILES_SERVER_URL', 'https://files.lalithadithyan.dev')
        urllib.request.urlretrieve(f"{files_url}/download/testing_mask_dataset.zip", zip_path)
        print("Unzipping testing mask dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("datasets")
        if os.path.exists(zip_path):
            os.remove(zip_path)
    else:
        print("Testing mask dataset already present.")

    # 3. Configure PlacesTraining Directory & config.yml
    run_path = "PlacesTraining"
    os.makedirs(run_path, exist_ok=True)
    config_path = os.path.join(run_path, "config.yml")
    
    if not os.path.exists(config_path):
        if os.path.exists("config.yml"):
            shutil.copy("config.yml", config_path)
        elif os.path.exists("checkpoints_places/config.yml"):
            shutil.copy("checkpoints_places/config.yml", config_path)
        else:
            files_url = os.environ.get('FILES_SERVER_URL', 'https://files.lalithadithyan.dev')
            urllib.request.urlretrieve(f"{files_url}/download/config.yml", config_path)

    # 4. Checkpoint weights restoration
    gen_dest = os.path.join(run_path, "InpaintingModel_gen.pth")
    dis_dest = os.path.join(run_path, "InpaintingModel_dis.pth")
    
    if os.path.exists(gen_dest) and os.path.exists(dis_dest):
        print(f"Local checkpoints already exist in {run_path}")
    else:
        files_url = os.environ.get('FILES_SERVER_URL', 'https://files.lalithadithyan.dev')
        session = os.environ.get('C2_SESSION', 'PLACES2_run')
        list_url = f"{files_url}/api/models?session={session}"
        print(f"Fetching checkpoint listing from: {list_url}")
        try:
            req = urllib.request.Request(list_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            file_list = data.get('files', []) if isinstance(data, dict) else data
            gen_files = [(int(m.group(1)), f) for f in file_list for m in [re.match(r'^DAVA_(\d{9})_InpaintingModel_gen\.pth$', f)] if m]
            dis_files = [(int(m.group(1)), f) for f in file_list for m in [re.match(r'^DAVA_(\d{9})_InpaintingModel_dis\.pth$', f)] if m]
            if gen_files and dis_files:
                common_iter = min(max(gen_files, key=lambda x: x[0])[0], max(dis_files, key=lambda x: x[0])[0])
                best_gen_name = [x[1] for x in gen_files if x[0] == common_iter][0]
                best_dis_name = [x[1] for x in dis_files if x[0] == common_iter][0]
                print(f"Downloading checkpoints at iteration {common_iter}...")
                urllib.request.urlretrieve(f"{files_url}/download/{best_gen_name}", gen_dest)
                urllib.request.urlretrieve(f"{files_url}/download/{best_dis_name}", dis_dest)
                print("[OK] Restored checkpoints from server.")
            else:
                print(f"No existing checkpoints found on server for session '{session}'. Starting fresh.")
        except Exception as e:
            print(f"Could not restore checkpoints from server: {e}")

    # 5. Update YAML Config for Places2 (FILTER_BY_SEG_MASK=True, MAX_CATEGORIES=53)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg['TRAIN_INPAINT_IMAGE_FLIST'] = 'datasets/places365/places365_standard/train'
    cfg['TEST_INPAINT_IMAGE_FLIST'] = 'datasets/places365/places365_standard/val'
    cfg['TRAIN_MASK_FLIST'] = 'datasets/testing_mask_dataset'
    cfg['TEST_MASK_FLIST'] = 'datasets/testing_mask_dataset'
    cfg['FILTER_BY_SEG_MASK'] = True
    max_cats = os.environ.get('MAX_CATEGORIES', '53')
    cfg['MAX_CATEGORIES'] = int(max_cats) if max_cats.isdigit() else 53
    print(f"Places2 config updated: FILTER_BY_SEG_MASK=True, MAX_CATEGORIES={cfg['MAX_CATEGORIES']}")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

def log_setup_run(start_time, status, error_msg=None):
    duration = time.time() - start_time
    m, s = divmod(int(duration), 60)
    h, m = divmod(m, 60)
    dur_str = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s" if m > 0 else f"{s}s"
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    session = os.environ.get('C2_SESSION', 'PLACES2_run')
    
    log_line = f"[{timestamp}] Session: {session} | Status: {status} | Duration: {dur_str}"
    if error_msg:
        clean_err = str(error_msg).replace('\n', ' ').strip()
        if len(clean_err) > 80:
            clean_err = clean_err[:77] + "..."
        log_line += f" | Error: {clean_err}"
    log_line += "\n"
    
    try:
        with open("setup_history.txt", "a") as f:
            f.write(log_line)
    except Exception as e:
        sys.stderr.write(f"Failed to write to setup_history.txt: {e}\n")


def main():
    start_time = time.time()
    
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        session = os.environ.get('C2_SESSION', 'PLACES2_run')
        with open("setup_history.txt", "a") as f:
            f.write(f"[{timestamp}] Session: {session} | Status: STARTED\n")
    except Exception:
        pass

    # Init live status reporter
    c2_url  = os.environ.get('C2_SERVER_URL', 'https://lalithadithyan.dev')
    session = os.environ.get('C2_SESSION', 'PLACES2_run')
    try:
        global _reporter
        _reporter = StatusReporter(c2_url, session)
        print(f"[Dashboard] Live status: {c2_url}/dashboard/status?session={session}")
    except Exception as e:
        print(f"[Dashboard] Reporter init failed (non-fatal): {e}")
        _reporter = None

    is_interactive = sys.stdout.isatty()
    if is_interactive:
        sys.stdout.write("\n")
        sys.stdout.flush()

    try:
        # Step 1: Verify Workspace
        execute_step(1, "Verifying workspace directory", step_verify_workspace)
        
        # Step 2: Set up CUDA Environment
        execute_step(2, "Configuring CUDA environment", step_setup_cuda)
        
        # Step 3: Create Venv
        execute_step(3, "Creating Python virtual environment", step_create_venv)
        
        # Step 4: Upgrade pip & wheel
        execute_step(4, "Upgrading pip and wheel", ["pip", "install", "--upgrade", "pip", "wheel"])
        
        # Step 5: Install setuptools < 82 & numpy < 2
        _STEP5 = "Installing setuptools < 82 and numpy < 2"
        if check_setuptools_numpy_installed():
            print(f"[Step 5/{TOTAL_STEPS}] setuptools < 82 and numpy < 2 already installed. Skipping.")
            if _reporter: _reporter.skip_step(5, _STEP5)
        else:
            execute_step(5, _STEP5, ["pip", "install", "setuptools<82", "numpy<2"])
        
        # Step 6: Install PyTorch 2.1.2
        _STEP6 = "Installing PyTorch 2.1.2 (CUDA 12.1 whl)"
        if check_pytorch_installed():
            print(f"[Step 6/{TOTAL_STEPS}] PyTorch 2.1.2 and torchvision 0.16.2 already installed. Skipping.")
            if _reporter: _reporter.skip_step(6, _STEP6)
        else:
            execute_step(6, _STEP6, [
                "pip", "install", "torch==2.1.2", "torchvision==0.16.2",
                "--extra-index-url", "https://download.pytorch.org/whl/cu121"
            ])

        # Step 7: Install packaging & ninja
        _STEP7 = "Installing packaging and ninja compiler tool"
        if check_ninja_packaging_installed():
            print(f"[Step 7/{TOTAL_STEPS}] packaging and ninja already installed. Skipping.")
            if _reporter: _reporter.skip_step(7, _STEP7)
        else:
            execute_step(7, _STEP7, ["pip", "install", "packaging", "ninja"])

        # Step 8: Compile causal-conv1d & mamba-ssm
        _STEP8 = "Compiling causal-conv1d and mamba-ssm (verbose)"
        if check_mamba_installed():
            print(f"[Step 8/{TOTAL_STEPS}] causal-conv1d and mamba-ssm already compiled and installed. Skipping.")
            if _reporter: _reporter.skip_step(8, _STEP8)
        else:
            execute_step(8, _STEP8, [
                "pip", "install", "causal-conv1d==1.1.3.post1", "mamba-ssm==1.1.3.post1",
                "--no-build-isolation", "-v"
            ])

        # Step 9: Install other requirements
        execute_step(9, "Installing remaining requirements.txt dependencies", ["pip", "install", "-r", "requirements.txt"])

        # Step 10: Download InternImage ops_dcnv3
        execute_step(10, "Downloading InternImage ops_dcnv3 folder", step_download_ops)

        # Step 11: Compile ops_dcnv3
        _STEP11 = "Compiling ops_dcnv3 CUDA kernels"
        if check_dcnv3_compiled():
            print(f"[Step 11/{TOTAL_STEPS}] ops_dcnv3 CUDA kernels already compiled and installed. Skipping.")
            if _reporter: _reporter.skip_step(11, _STEP11)
        else:
            ops_dir = os.path.abspath(os.path.join("src", "ops_dcnv3"))
            make_sh = os.path.join(ops_dir, "make.sh")
            try:
                os.chmod(make_sh, 0o755)
            except Exception:
                pass
            patch_pytorch_boxing_header()
            build_dir = os.path.join(ops_dir, "build")
            if os.path.exists(build_dir):
                print(f"Cleaning existing build directory: {build_dir}")
                shutil.rmtree(build_dir, ignore_errors=True)
            execute_step(11, _STEP11, ["make.sh"], cwd=ops_dir, shell=True)

        # Step 12: Places2 Dataset & Config Setup
        execute_step(12, "Setting up Places2 dataset & PlacesTraining config", step_setup_places_dataset_and_config)

    except BaseException as e:
        err_msg = str(e) or type(e).__name__
        log_setup_run(start_time, "FAILED", err_msg)
        sys.exit(1)

    if is_interactive:
        sys.stdout.write("\n")

    print("\n==========================================================")
    print("      SEM-Net Places2 Setup Completed Successfully!        ")
    print("==========================================================")
    print("Launching training script: main.py --path PlacesTraining")
    print("==========================================================")

    log_setup_run(start_time, "SUCCESS")

    if "--no-launch" in sys.argv or "--setup-only" in sys.argv:
        print("Setup completed. Skipping auto-launch of training script.")
        sys.exit(0)

    # Determine Python executable and launch command
    venv_python = os.path.abspath(os.path.join("venv", "bin", "python"))
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable

    # Detect GPU count for DDP vs single GPU execution
    try:
        out = subprocess.check_output(
            [python_bin, "-c", "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"],
            text=True
        ).strip()
        num_gpus = int(out)
    except Exception:
        num_gpus = 1

    run_path = "PlacesTraining"

    if num_gpus > 1:
        venv_torchrun = os.path.abspath(os.path.join("venv", "bin", "torchrun"))
        torchrun_bin = venv_torchrun if os.path.exists(venv_torchrun) else shutil.which("torchrun") or "torchrun"
        cmd = [
            torchrun_bin,
            f"--nproc_per_node={num_gpus}",
            "--master_port=29501",
            "main.py",
            "--model", "2",
            "--path", run_path
        ]
        print(f"Launching Multi-GPU DDP Places2 Training ({num_gpus} GPUs): {' '.join(cmd)}")
        sys.stdout.flush()
        os.execv(torchrun_bin, cmd)
    else:
        cmd = [
            python_bin,
            "main.py",
            "--model", "2",
            "--path", run_path
        ]
        print(f"Launching Single-GPU Places2 Training: {' '.join(cmd)}")
        sys.stdout.flush()
        os.execv(python_bin, cmd)

if __name__ == "__main__":
    main()
