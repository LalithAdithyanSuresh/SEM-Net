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

TOTAL_STEPS = 14

# ── Live Status Reporter ───────────────────────────────────────────────────
# Posts live step progress to the C2 server for the dashboard.
# Uses only stdlib (no pip deps), fires-and-forgets via a queued worker.

_reporter = None  # initialized in main()

class StatusReporter:
    def __init__(self, c2_url, session):
        self.c2_url     = c2_url.rstrip('/')
        self.session    = session
        self.started_at = time.time()
        self.host       = socket.gethostname()
        self._lock      = threading.Lock()
        self.steps      = {
            i: {'num': i, 'name': STEP_NAMES_DEFAULT[i-1], 'status': 'pending',
                'started_at': None, 'finished_at': None, 'duration': None, 'logs': []}
            for i in range(1, TOTAL_STEPS + 1)
        }
        self._queue     = queue.Queue()
        # Start a single worker thread to serialize updates and enforce delays
        threading.Thread(target=self._worker, daemon=True).start()
        self._post(self._payload())  # register immediately

    # ── Default step names (populated before setup_server knows them) ─
    STEP_NAMES_DEFAULT = [
        'Verifying workspace directory',
        'Configuring CUDA 12.4 environment',
        'Creating Python virtual environment',
        'Upgrading pip and wheel',
        'Installing setuptools < 82 and numpy < 2',
        'Installing PyTorch 2.1.2',
        'Installing packaging and ninja',
        'Compiling causal-conv1d and mamba-ssm',
        'Installing remaining requirements.txt',
        'Downloading InternImage ops_dcnv3',
        'Compiling ops_dcnv3 CUDA kernels',
        'Download & extract mask dataset',
        'Download & extract Places365 dataset',
        'Restoring latest model checkpoint',
    ]

    def _read_history(self):
        try:
            if not os.path.exists("setup_history.txt"):
                return []
            with open("setup_history.txt", "r") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
                return lines[-50:]
        except Exception:
            return []

    def _payload(self):
        with self._lock:
            return {
                'session':    self.session,
                'host':       self.host,
                'started_at': self.started_at,
                'phase':      'setup',
                'total_steps': TOTAL_STEPS,
                'steps':      list(self.steps.values()),
                'history':    self._read_history(),
            }

    def _post(self, payload):
        """Enqueue payload to be processed by background worker."""
        self._queue.put(payload)

    def _worker(self):
        """Processes enqueued payloads sequentially, enforcing a 2s delay after sending."""
        global _printed_warning
        context = ssl._create_unverified_context()
        while True:
            payload = self._queue.get()
            try:
                body = json.dumps(payload).encode()
                req  = urllib.request.Request(
                    f"{self.c2_url}/api/setup_status", data=body,
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    }, method='POST')
                urllib.request.urlopen(req, timeout=5, context=context)
            except Exception as e:
                if not _printed_warning:
                    sys.stderr.write(f"\n[Dashboard Warning] Failed to post status to {self.c2_url}: {e}\n")
                    sys.stderr.flush()
                    _printed_warning = True
            finally:
                self._queue.task_done()

    def join(self):
        """Block until all queued status posts have been sent."""
        self._queue.join()

    def _flush(self):
        self._post(self._payload())

    def start_step(self, num, name):
        with self._lock:
            s = self.steps[num]
            s['name'] = name;  s['status'] = 'running'
            s['started_at'] = time.time();  s['logs'] = []
        self._flush()

    def add_log(self, num, line):
        with self._lock:
            logs = self.steps[num]['logs']
            logs.append(line)
            if len(logs) > 500:
                self.steps[num]['logs'] = logs[-500:]
            count = len(self.steps[num]['logs'])
        if count % 8 == 0:   # post every 8 lines to avoid flooding
            self._flush()

    def complete_step(self, num):
        t = time.time()
        with self._lock:
            s = self.steps[num]
            s['status'] = 'done';  s['finished_at'] = t
            s['duration'] = t - (s['started_at'] or t)
        self._flush()

    def skip_step(self, num, name):
        t = time.time()
        with self._lock:
            s = self.steps[num]
            s['name'] = name;  s['status'] = 'skipped'
            s['started_at'] = t;  s['finished_at'] = t
            s['duration'] = 0
            s['logs'] = ['Already installed / exists. Skipped.']
        self._flush()

    def fail_step(self, num, error):
        t = time.time()
        with self._lock:
            s = self.steps[num]
            s['status'] = 'error';  s['finished_at'] = t
            s['duration'] = t - (s['started_at'] or t)
            s['logs'].append(f'ERROR: {error}')
        self._flush()

STEP_NAMES_DEFAULT = StatusReporter.STEP_NAMES_DEFAULT if False else [
    'Verifying workspace directory',
    'Configuring CUDA 12.4 environment',
    'Creating Python virtual environment',
    'Upgrading pip and wheel',
    'Installing setuptools < 82 and numpy < 2',
    'Installing PyTorch 2.1.2',
    'Installing packaging and ninja',
    'Compiling causal-conv1d and mamba-ssm',
    'Installing remaining requirements.txt',
    'Downloading InternImage ops_dcnv3',
    'Compiling ops_dcnv3 CUDA kernels',
    'Download & extract mask dataset',
    'Download & extract Places365 dataset',
    'Restoring latest model checkpoint',
]


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
    global _reporter
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
            if _reporter:
                _reporter.add_log(step_num, line_str)
            if is_interactive:
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
    global _reporter
    is_interactive = sys.stdout.isatty()

    if _reporter:
        _reporter.start_step(step_num, step_name)

    if is_interactive:
        draw_progress_bar(step_num, step_name, TOTAL_STEPS)
    else:
        print(f"\n=== [Step {step_num}/{TOTAL_STEPS}] {step_name} ===")
        sys.stdout.flush()

    try:
        if callable(func_or_cmd):
            # Intercept print output so callable steps also get logged to dashboard
            if _reporter:
                _orig_write = sys.stdout.write
                def _cap_write(text):
                    _orig_write(text)
                    for ln in text.splitlines():
                        if ln.strip():
                            _reporter.add_log(step_num, ln)
                sys.stdout.write = _cap_write
                try:
                    func_or_cmd()
                finally:
                    sys.stdout.write = _orig_write
            else:
                func_or_cmd()
            if is_interactive:
                draw_progress_bar(step_num, step_name, TOTAL_STEPS)
        else:
            cmd = list(func_or_cmd)
            if cmd and cmd[0] == "pip":
                cmd[0] = get_pip_executable()
            run_command(cmd, step_num, step_name, TOTAL_STEPS, shell=shell, cwd=cwd)

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
        print(f"WARNING: CUDA 12.4 Toolkit directory not found at {cuda_dir}. Skipping CUDA configuration.")
        return
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
        
    url = "https://files.lalithadithyan.dev/download/places365standard_easyformat.tar"
    archive_path = os.path.join("datasets", "places365standard_easyformat.tar")
    
    download_file(url, archive_path)
    extract_tar(archive_path, dest_dir)
    
    if os.path.exists(archive_path):
        os.remove(archive_path)

def step_download_latest_model():
    """Query the C2 files server, find the highest-iteration checkpoint pair
    (gen + dis) for the active session, and download them to PlacesTraining/."""
    files_url = os.environ.get("FILES_SERVER_URL", "https://files.lalithadithyan.dev")
    session   = os.environ.get("C2_SESSION", "DAVA")
    run_path  = "./PlacesTraining"

    gen_dest = os.path.join(run_path, "InpaintingModel_gen.pth")
    dis_dest = os.path.join(run_path, "InpaintingModel_dis.pth")

    # ── 1. Fetch the directory listing from the files server ──────────────
    list_url = f"{files_url}/api/models?session={session}"
    print(f"Fetching model listing from: {list_url}")
    try:
        req = urllib.request.Request(
            list_url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"WARNING: Could not reach files server ({e}). Skipping model download.")
        return

    # ── 2. Parse the file list – accept both {"files": [...]} and plain list ─
    if isinstance(data, dict):
        file_list = data.get("files", [])
    elif isinstance(data, list):
        file_list = data
    else:
        print("WARNING: Unexpected response format from files server. Skipping.")
        return

    if not file_list:
        print(f"No checkpoint files found on server for session '{session}'. Starting fresh.")
        return

    # ── 3. Find the highest-iteration gen and dis checkpoints ────────────────
    # Filename pattern:  DAVA_000002000_InpaintingModel_gen.pth
    import re
    gen_files = [(int(m.group(1)), f) for f in file_list
                 for m in [re.match(r'^DAVA_(\d{9})_InpaintingModel_gen\.pth$', f)] if m]
    dis_files = [(int(m.group(1)), f) for f in file_list
                 for m in [re.match(r'^DAVA_(\d{9})_InpaintingModel_dis\.pth$', f)] if m]

    if not gen_files or not dis_files:
        print("No DAVA checkpoint files found on server. Starting fresh.")
        return

    best_iter_gen, best_gen_name = max(gen_files, key=lambda x: x[0])
    best_iter_dis, best_dis_name = max(dis_files, key=lambda x: x[0])

    if best_iter_gen != best_iter_dis:
        # Prefer the lower of the two so both gen+dis come from the same save
        common_iter = min(best_iter_gen, best_iter_dis)
        candidates_gen = [x for x in gen_files if x[0] == common_iter]
        candidates_dis = [x for x in dis_files if x[0] == common_iter]
        if not candidates_gen or not candidates_dis:
            print(f"WARNING: Could not find matching gen+dis pair at iteration {common_iter}. Skipping.")
            return
        best_gen_name = candidates_gen[0][1]
        best_dis_name = candidates_dis[0][1]
        best_iter_gen = common_iter

    print(f"Latest checkpoint pair found at iteration {best_iter_gen:,}:")
    print(f"  gen -> {best_gen_name}")
    print(f"  dis -> {best_dis_name}")

    os.makedirs(run_path, exist_ok=True)

    # ── 4. Download gen ──────────────────────────────────────────────────────
    # Files server stores everything flat: /download/<filename>
    gen_url = f"{files_url}/download/{best_gen_name}"
    print(f"Downloading generator checkpoint...")
    download_file(gen_url, gen_dest)

    # ── 5. Download dis ──────────────────────────────────────────────────────
    dis_url = f"{files_url}/download/{best_dis_name}"
    print(f"Downloading discriminator checkpoint...")
    download_file(dis_url, dis_dest)

    print(f"[OK] Model checkpoints restored to {run_path}/ (iteration {best_iter_gen:,})")


def log_setup_run(start_time, status, error_msg=None):
    duration = time.time() - start_time
    m, s = divmod(int(duration), 60)
    h, m = divmod(m, 60)
    dur_str = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s" if m > 0 else f"{s}s"
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    session = os.environ.get('C2_SESSION', 'DAVA')
    
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
    
    # Flush status reporter so final run state is recorded on C2
    if _reporter:
        try:
            _reporter._flush()
            _reporter.join()
        except Exception:
            pass


def main():
    global _reporter
    start_time = time.time()

    # Log start to history file
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        session = os.environ.get('C2_SESSION', 'DAVA')
        with open("setup_history.txt", "a") as f:
            f.write(f"[{timestamp}] Session: {session} | Status: STARTED\n")
    except Exception:
        pass

    # Init live status reporter (posts to C2 server in background)
    c2_url  = os.environ.get('C2_SERVER_URL', 'https://lalithadithyan.dev')
    session = os.environ.get('C2_SESSION', 'DAVA')
    try:
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
        
        # Step 2: Set up CUDA 12.4 Environment
        execute_step(2, "Configuring CUDA 12.4 environment", step_setup_cuda)
        
        # Step 3: Create Venv
        execute_step(3, "Creating Python virtual environment", step_create_venv)
        
        # Step 4: Upgrade pip & wheel
        execute_step(4, "Upgrading pip and wheel", ["pip", "install", "--upgrade", "pip", "wheel"])
        
        # Step 5: Install compatible setuptools & numpy pins
        _STEP5 = "Installing setuptools < 82 and numpy < 2"
        if check_setuptools_numpy_installed():
            print(f"[Step 5/{TOTAL_STEPS}] setuptools < 82 and numpy < 2 already installed. Skipping.")
            if _reporter: _reporter.skip_step(5, _STEP5)
        else:
            execute_step(5, _STEP5, ["pip", "install", "setuptools<82", "numpy<2"])
        
        # Step 6: Install PyTorch 2.1.2 (compatible with CUDA 12.4 compiler)
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
            execute_step(11, _STEP11, ["sh", "make.sh"], cwd=ops_dir)

        # Step 12: Download & extract mask dataset
        execute_step(12, "Download & extract mask dataset", step_download_mask_dataset)
        
        # Step 13: Download & extract Places365 dataset
        execute_step(13, "Download & extract Places365 dataset", step_download_places365_dataset)

        # Step 14: Download latest model checkpoint from C2 files server
        execute_step(14, "Restoring latest model checkpoint from files server", step_download_latest_model)

    except BaseException as e:
        err_msg = str(e) or type(e).__name__
        log_setup_run(start_time, "FAILED", err_msg)
        sys.exit(1)

    if is_interactive:
        sys.stdout.write("\n")

    print("\n==========================================================")
    print("          SEM-Net Setup Completed Successfully!           ")
    print("==========================================================")
    print("Launching training script: run_training_c2.sh")
    print("==========================================================")

    log_setup_run(start_time, "SUCCESS")

    # Auto-launch the training script
    training_script = os.path.abspath("run_training_c2.sh")
    session_name = os.environ.get("C2_SESSION", "DAVA")

    if not os.path.exists(training_script):
        print(f"ERROR: Training script not found at {training_script}. Please run it manually.")
        sys.exit(1)

    try:
        os.chmod(training_script, 0o755)
    except Exception:
        pass

    print(f"Running: bash {training_script} {session_name}")
    sys.stdout.flush()
    os.execv("/bin/bash", ["/bin/bash", training_script, session_name])

if __name__ == "__main__":
    main()
