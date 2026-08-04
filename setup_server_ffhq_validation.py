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

    # ── Default step names (populated before setup_server knows them) ─
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
        'Generating FastSAM segment masks (train_seg & test_seg)',
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
            # Safely extract last few lines of log/history for status context
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
                # Keep C2 status posting lightweight, ignore errors silently
                url = f"{self.c2_url}/api/setup/status"
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode('utf-8'),
                    headers={'Content-Type': 'application/json', 'User-Agent': 'StatusReporter/1.0'}
                )
                # Create ssl context ignoring validation if needed
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
    # Retrieve active virtual env python/pip path if exists
    def get_pip_executable():
        venv_pip = os.path.abspath(os.path.join("venv", "bin", "pip"))
        if os.path.exists(venv_pip):
            return venv_pip
        return "pip"

    def run_command(cmd, cwd=None, shell=False):
        print(f"Executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        sys.stdout.flush()
        # Ensure correct environment is propagated
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
        
        # Stream output in real-time
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
        raise RuntimeError(f"'nvcc' not found or failed: {e}. CUDA compilation support is required.")

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
        raise RuntimeError("Could not find boxing.h header file in PyTorch site-packages for ops_dcnv3 patching.")
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

def step_generate_segment_masks():
    venv_python = os.path.abspath(os.path.join("venv", "bin", "python"))
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable
    script_path = os.path.abspath("generate_segment_masks.py")
    if not os.path.exists(script_path):
        raise RuntimeError("generate_segment_masks.py script not found!")
    print("Running FastSAM segment mask generator across available GPUs (batch size 16)...")
    subprocess.check_call([python_bin, script_path, "--batch-size", "16"])

def log_setup_run(start_time, status, error_msg=None):
    duration = time.time() - start_time
    m, s = divmod(int(duration), 60)
    h, m = divmod(m, 60)
    dur_str = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s" if m > 0 else f"{s}s"
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    session = os.environ.get('C2_SESSION', 'validation')
    
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
        session = os.environ.get('C2_SESSION', 'validation')
        with open("setup_history.txt", "a") as f:
            f.write(f"[{timestamp}] Session: {session} | Status: STARTED\n")
    except Exception:
        pass

    # Init live status reporter
    c2_url  = os.environ.get('C2_SERVER_URL', 'https://lalithadithyan.dev')
    session = os.environ.get('C2_SESSION', 'validation')
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
        
        # Step 5: Install compatible setuptools & numpy pins
        _STEP5 = "Installing setuptools < 82 and numpy < 2"
        if check_setuptools_numpy_installed():
            print(f"[Step 5/{TOTAL_STEPS}] setuptools < 82 and numpy < 2 already installed. Skipping.")
            if _reporter: _reporter.skip_step(5, _STEP5)
        else:
            execute_step(5, _STEP5, ["pip", "install", "setuptools<82", "numpy<2"])
        
        # Step 6: Install PyTorch 2.1.2 (compatible with CUDA compiler)
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

        # Step 12: Generate FastSAM segment masks
        execute_step(12, "Generating FastSAM segment masks (train_seg & test_seg)", step_generate_segment_masks)

    except BaseException as e:
        err_msg = str(e) or type(e).__name__
        log_setup_run(start_time, "FAILED", err_msg)
        sys.exit(1)

    if is_interactive:
        sys.stdout.write("\n")

    print("\n==========================================================")
    print("          SEM-Net Setup Completed Successfully!           ")
    print("==========================================================")
    print("Launching evaluation script: evaluate_ffhq.py")
    print("==========================================================")

    log_setup_run(start_time, "SUCCESS")

    if "--no-launch" in sys.argv or "--setup-only" in sys.argv:
        print("Setup completed. Skipping auto-launch of evaluation script.")
        sys.exit(0)

    # Auto-launch the evaluation script
    eval_script = os.path.abspath("evaluate_ffhq.py")
    venv_python = os.path.abspath(os.path.join("venv", "bin", "python"))
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable

    if not os.path.exists(eval_script):
        print(f"ERROR: Evaluation script not found at {eval_script}. Please run it manually.")
        sys.exit(1)

    eval_cmd = [python_bin, eval_script]
    
    # Filter out script arguments and pass options to evaluate_ffhq.py
    passed_args = [arg for arg in sys.argv[1:] if arg not in ("--no-launch", "--setup-only")]
    eval_cmd.extend(passed_args)

    print(f"Running: {' '.join(eval_cmd)}")
    sys.stdout.flush()
    
    # Run evaluate_ffhq.py using execv
    os.execv(python_bin, eval_cmd)

if __name__ == "__main__":
    main()
