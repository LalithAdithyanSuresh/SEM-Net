#!/bin/bash

# Automatically switch to the script's directory
cd "$(dirname "$0")"

# Check if we need to locate and transition to the repository root
if [ -z "$SEMNET_LAUNCHED_FROM_REPO" ]; then
    if [ ! -f "main.py" ] || [ ! -f "push_logs.py" ]; then
        echo "[*] main.py or push_logs.py not found in current directory ($(pwd)). Searching in subdirectories..."
        FOUND_DIR=$(find . -maxdepth 3 -name "main.py" -exec dirname {} \; | head -n 1)
        if [ -n "$FOUND_DIR" ] && [ -f "$FOUND_DIR/push_logs.py" ]; then
            LAUNCH_DIR="$(pwd)"
            # Get absolute path of repository
            cd "$FOUND_DIR"
            REPO_DIR="$(pwd)"
            cd "$LAUNCH_DIR"
            
            echo "[*] Found repository root at: $REPO_DIR"
            
            # Helper to move directories safely
            move_dir_safely() {
                local src="$1"
                local dest="$2"
                if [ -d "$src" ]; then
                    echo "[*] Moving contents of $src to $dest..."
                    mkdir -p "$dest"
                    cp -r "$src"/. "$dest/" 2>/dev/null || true
                    rm -rf "$src"
                fi
            }
            
            move_dir_safely "dataset" "$REPO_DIR/dataset"
            move_dir_safely "checkpoints_ffhq" "$REPO_DIR/checkpoints_ffhq"
            
            if [ -f "$REPO_DIR/setup_ffhq_training.sh" ]; then
                echo "[*] Executing repository version of setup_ffhq_training.sh..."
                export SEMNET_LAUNCHED_FROM_REPO=1
                exec bash "$REPO_DIR/setup_ffhq_training.sh" "$@"
            else
                echo "[*] Changing directory to repository root: $REPO_DIR"
                cd "$REPO_DIR"
            fi
        else
            echo "[ERROR] Could not find the repository directory containing main.py and push_logs.py."
            echo "Please run this script from within the SEM-Net repository."
            exit 1
        fi
    fi
fi

# Configuration variables
export C2_SERVER_URL="https://lalithadithyan.dev"
export FILES_SERVER_URL="https://files.lalithadithyan.dev"
export VALIDATION_SERVER_URL="https://validate.lalithadithyan.dev"
export C2_SESSION="FFHQ_run"
export TORCH_HOME="./tmp/torch_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================="
echo "FFHQ Fine-Tuning Setup and Training Script"
echo "================================================="

# Create datasets directory if not exists
mkdir -p datasets
mkdir -p dataset

# Set CUDA Toolkit paths (critical for compiling causal-conv1d, mamba-ssm, and DCNv3)
export CUDA_HOME="/usr/local/cuda-12.4"
if [ ! -d "$CUDA_HOME" ]; then
    export CUDA_HOME="/usr/local/cuda"
fi
if [ -d "$CUDA_HOME" ]; then
    echo "[*] Configuring CUDA environment at $CUDA_HOME..."
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
else
    echo "[WARNING] CUDA Toolkit directory not found. Package compilation might fail."
fi

# 1. Detect, create, and activate virtual environment
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
    echo "[*] No virtual environment found. Creating one (.venv)..."
    python3 -m venv .venv
fi

if [ -d ".venv" ]; then
    echo "[*] Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "[*] Activating virtual environment (venv)..."
    source venv/bin/activate
fi

# 2. Install required pip packages (ensures PyTorch is compiled for CUDA 12.1)
echo "[*] Installing dependencies..."
pip install torch==2.1.2 torchvision==0.16.2 --extra-index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0.0" omegaconf webdataset pytorch-lightning kornia joblib hydra-core requests scikit-image easydict opencv-python tabulate scikit-learn pyyaml pandas matplotlib packaging einops timm gdown

# 2.5 Compile and install Mamba modules if missing
if ! python -c "import mamba_ssm" &>/dev/null; then
    echo "[*] mamba_ssm not found. Compiling and installing causal-conv1d and mamba-ssm..."
    pip install causal-conv1d==1.1.3.post1 mamba-ssm==1.1.3.post1 --no-build-isolation -v
else
    echo "[*] mamba_ssm already installed."
fi

# 2.6 Compile DCNv3 CUDA kernels if missing
if ! python -c "import DCNv3" &>/dev/null; then
    echo "[*] DCNv3 not found. Compiling DCNv3 CUDA kernels..."
    # Apply boxing.h patch to ATen if it exists in PyTorch
    python -c "
import glob, os
paths = glob.glob(os.path.join('.venv', 'lib', 'python*', 'site-packages', 'torch', 'include', 'ATen', 'core', 'boxing', 'impl', 'boxing.h')) + \
        glob.glob(os.path.join('venv', 'lib', 'python*', 'site-packages', 'torch', 'include', 'ATen', 'core', 'boxing', 'impl', 'boxing.h'))
for path in paths:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        if '// has_ivalue_to<T> tests the presence' in content and 'struct has_ivalue_to : std::true_type {};' not in content:
            print(f'Patching ATen boxing.h: {path}')
            start_idx = content.find('// has_ivalue_to<T> tests')
            end_idx = content.find('// boxing predicates')
            rep = '// has_ivalue_to<T> tests the presence/absence of instance method IValue::to<T>()\n//\ntemplate <class T, class Enable = void>\nstruct has_ivalue_to : std::true_type {};\n\n'
            new_content = content[:start_idx] + rep + content[end_idx:]
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
    except Exception as e:
        print(f'Failed to patch: {e}')
"
    
    cd src/ops_dcnv3
    chmod +x make.sh
    rm -rf build/
    sh make.sh
    cd ../..
else
    echo "[*] DCNv3 already compiled."
fi

# 3. Synchronize branch to LAST_RUN
echo "[*] Pulling latest changes from branch LAST_RUN..."
git fetch origin
git checkout LAST_RUN
git pull origin LAST_RUN

# 4.3 Masks Dataset
if [ ! -d "datasets/testing_mask_dataset" ]; then
    echo "[*] Downloading testing mask dataset..."
    curl -L -o datasets/testing_mask_dataset.zip https://files.lalithadithyan.dev/download/testing_mask_dataset.zip
    echo "[*] Unzipping testing mask dataset..."
    unzip datasets/testing_mask_dataset.zip -d datasets/
    rm datasets/testing_mask_dataset.zip
else
    echo "[*] Testing mask dataset already exists."
fi

# 5. Checkpoint management (Start vs. Continue training)
MODE=${1:-"continue"}
CHECKPOINT_DIR="checkpoints_ffhq"

if [ "$MODE" == "start" ]; then
    if [ -d "$CHECKPOINT_DIR" ]; then
        BACKUP_DIR="${CHECKPOINT_DIR}_backup_$(date +%s)"
        echo "[*] Start mode selected. Renaming existing checkpoints directory to $BACKUP_DIR..."
        mv "$CHECKPOINT_DIR" "$BACKUP_DIR"
    fi
    mkdir -p "$CHECKPOINT_DIR"
    echo "[*] Downloading default model configuration config.yml..."
    gdown 1tXf-AhpM9To83fVIrAaPJ82pO3ant_7V -O "$CHECKPOINT_DIR/config.yml"
else
    echo "[*] Continue mode selected (resuming from pre-trained weights)."
    mkdir -p "$CHECKPOINT_DIR"
    
    # Download generator weights if missing
    if [ ! -f "$CHECKPOINT_DIR/InpaintingModel_gen.pth" ]; then
        echo "[*] Downloading pre-trained generator weights from Google Drive..."
        gdown 1Pr4mg3qo2zlGtEI9GMdFMmGkU4V5kpIi -O "$CHECKPOINT_DIR/InpaintingModel_gen.pth"
    fi
    
    # Verify generator file corruption
    if [ -f "$CHECKPOINT_DIR/InpaintingModel_gen.pth" ] && head -n 1 "$CHECKPOINT_DIR/InpaintingModel_gen.pth" | grep -q "^<"; then
        echo "[!] Generator weights file appears to be a corrupted HTML file. Re-downloading..."
        rm -f "$CHECKPOINT_DIR/InpaintingModel_gen.pth"
        gdown 1Pr4mg3qo2zlGtEI9GMdFMmGkU4V5kpIi -O "$CHECKPOINT_DIR/InpaintingModel_gen.pth"
    fi

    # Download discriminator weights if missing
    if [ ! -f "$CHECKPOINT_DIR/InpaintingModel_dis.pth" ]; then
        echo "[*] Downloading pre-trained discriminator weights from Google Drive..."
        gdown 116S6kNiocQH6v7l_0wdI7qtnjUbDx8rv -O "$CHECKPOINT_DIR/InpaintingModel_dis.pth"
    fi
    
    # Verify discriminator file corruption
    if [ -f "$CHECKPOINT_DIR/InpaintingModel_dis.pth" ] && head -n 1 "$CHECKPOINT_DIR/InpaintingModel_dis.pth" | grep -q "^<"; then
        echo "[!] Discriminator weights file appears to be a corrupted HTML file. Re-downloading..."
        rm -f "$CHECKPOINT_DIR/InpaintingModel_dis.pth"
        gdown 116S6kNiocQH6v7l_0wdI7qtnjUbDx8rv -O "$CHECKPOINT_DIR/InpaintingModel_dis.pth"
    fi

    # Download config if missing
    if [ ! -f "$CHECKPOINT_DIR/config.yml" ]; then
        echo "[*] Downloading model configuration config.yml..."
        gdown 1tXf-AhpM9To83fVIrAaPJ82pO3ant_7V -O "$CHECKPOINT_DIR/config.yml"
    fi
fi

# 6. Dynamically update config settings for Single GPU 0 and local datasets path
echo "[*] Auto-configuring config.yml settings (GPU: [0], Local Dataset paths)..."
python -c "
import yaml
config_path = '$CHECKPOINT_DIR/config.yml'
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
cfg['TRAIN_INPAINT_IMAGE_FLIST'] = 'dataset/ffhq/train'
cfg['TEST_INPAINT_IMAGE_FLIST'] = 'dataset/ffhq/test'
cfg['TRAIN_MASK_FLIST'] = 'datasets/testing_mask_dataset'
cfg['TEST_MASK_FLIST'] = 'datasets/testing_mask_dataset'
cfg['GPU'] = [0]
with open(config_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"

# 7. Start the active training loop
echo "================================================="
echo "Starting SEM-Net Training loop: [$C2_SESSION] on GPU 0"
echo "================================================="

while true; do
    echo "Checking C2 Server status for [$C2_SESSION]..."
    
    # Check for custom remote shell commands
    python -c "
import requests, subprocess, os
try:
    url = os.environ.get('C2_SERVER_URL')
    sess = os.environ.get('C2_SESSION')
    res = requests.get(f'{url}/api/pop_shell_command', params={'session': sess}, timeout=5)
    if res.status_code == 200:
        shell_cmd = res.json().get('shell_command')
        if shell_cmd:
            print(f'\n[C2 REMOTE COMMAND] Executing: {shell_cmd}')
            proc = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True, timeout=30)
            output = (proc.stdout + '\n' + proc.stderr).strip()
            print(output)
            requests.post(f'{url}/api/logs', json={'lines': [f'[REMOTE OUTPUT] {l}' for l in output.split('\n')], 'session': sess}, timeout=5)
except Exception: pass
" 2>/dev/null

    # Check if C2 is telling us to stop or run
    CMD=$(python -c "import requests, os; url=os.environ.get('C2_SERVER_URL'); sess=os.environ.get('C2_SESSION'); print(requests.get(f'{url}/api/command', params={'session': sess}, timeout=5).json().get('command', 'run'))" 2>/dev/null)
    
    if [ "$CMD" == "stop" ]; then
        echo "C2 status is 'STOP'. Waiting for 'run' command..."
        sleep 5
        continue
    fi

    # Launch Single-GPU training and pipe output to C2
    python -u main.py --model 2 --path "$CHECKPOINT_DIR" 2>&1 | python -u push_logs.py
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 42 ]; then
        echo "Received restart signal. Syncing latest repository modifications..."
        git pull origin LAST_RUN
        sleep 2
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "Training session finished/stopped normally."
        sleep 5
    else
        echo "Training process exited with error code $EXIT_CODE. Re-attempting startup in 10s..."
        sleep 10
    fi
done
