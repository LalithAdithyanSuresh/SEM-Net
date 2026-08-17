#!/bin/bash

# Automatically switch to the script's directory
cd "$(dirname "$0")"

# Check if we need to locate and transition to the repository root
if [ -z "$SEMNET_LAUNCHED_FROM_REPO" ]; then
    if [ ! -f "main.py" ] || [ ! -f "push_logs.py" ]; then
        echo "[*] main.py or push_logs.py not found in current directory ($(pwd)). Searching..."
        FOUND_DIR=$(find . -maxdepth 3 -name "main.py" -exec dirname {} \; | head -n 1)
        if [ -n "$FOUND_DIR" ] && [ -f "$FOUND_DIR/push_logs.py" ]; then
            LAUNCH_DIR="$(pwd)"
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
            move_dir_safely "datasets" "$REPO_DIR/datasets"
            move_dir_safely "PlacesTraining" "$REPO_DIR/PlacesTraining"
            
            if [ -f "$REPO_DIR/setup_places2_training.sh" ]; then
                echo "[*] Executing repository version of setup_places2_training.sh..."
                export SEMNET_LAUNCHED_FROM_REPO=1
                exec bash "$REPO_DIR/setup_places2_training.sh" "$@"
            else
                echo "[*] Changing directory to repository root: $REPO_DIR"
                cd "$REPO_DIR"
            fi
        else
            echo "[ERROR] Could not find the repository directory containing main.py."
            exit 1
        fi
    fi
fi

# Configuration variables
export C2_SERVER_URL="https://lalithadithyan.dev"
export FILES_SERVER_URL="https://files.lalithadithyan.dev"
export VALIDATION_SERVER_URL="https://validate.lalithadithyan.dev"
export C2_SESSION="PLACES2_run"
export TORCH_HOME="./tmp/torch_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================="
echo "Places2 Fine-Tuning Setup and Training Script"
echo "================================================="

# Create datasets directory if not exists
mkdir -p datasets

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

# 2. Install required pip packages
echo "[*] Installing dependencies..."
pip install torch==2.1.2 torchvision==0.16.2 --extra-index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0.0" omegaconf webdataset pytorch-lightning kornia joblib hydra-core requests scikit-image easydict opencv-python tabulate scikit-learn pyyaml pandas matplotlib einops timm gdown

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
            rep = '// has_ivalue_to<T> tests the presence/absence of instance method IValue::to<T>()\\n//\\ntemplate <class T, class Enable = void>\\nstruct has_ivalue_to : std::true_type {};\\n\\n'
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

# 3. Synchronize active branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "[*] Pulling latest changes from active branch $CURRENT_BRANCH..."
git fetch origin
git checkout "$CURRENT_BRANCH"
git pull origin "$CURRENT_BRANCH"

# 4. Locate or Download Places365 dataset
PLACES_DIR="datasets/places365"
if [ ! -d "$PLACES_DIR" ]; then
    SERVER_DATASET_DIR="/tmp/cks/CAMINO-DAVA/SEM-Net/datasets/places365"
    if [ ! -d "$SERVER_DATASET_DIR" ]; then
        SERVER_DATASET_DIR="/mnt/datadrive/inpaint/places2/places365standard_easyformat"
    fi
    if [ -d "$SERVER_DATASET_DIR" ]; then
        echo "[*] Found Places365 dataset at $SERVER_DATASET_DIR. Creating symbolic link..."
        mkdir -p datasets
        ln -s "$SERVER_DATASET_DIR" "$PLACES_DIR"
    else
        echo "[*] Places365 dataset not found locally. Downloading from files server..."
        mkdir -p datasets
        curl -L -o datasets/places365standard_easyformat.tar https://files.lalithadithyan.dev/download/places365standard_easyformat.tar
        echo "[*] Extracting Places365 dataset..."
        tar -xf datasets/places365standard_easyformat.tar -C datasets/
        if [ -d "datasets/places365standard_easyformat" ] && [ ! -d "datasets/places365" ]; then
            mv datasets/places365standard_easyformat datasets/places365
        fi
        rm datasets/places365standard_easyformat.tar
    fi
else
    echo "[*] Places365 dataset already exists."
fi

# 4.3 Masks Dataset
if [ ! -d "datasets/testing_mask_dataset" ]; then
    echo "[*] Downloading testing mask dataset..."
    curl -L -o datasets/testing_mask_dataset.zip https://files.lalithadithyan.dev/download/testing_mask_dataset.zip
    echo "[*] Unzipping testing mask dataset..."
    unzip -q datasets/testing_mask_dataset.zip -d datasets/
    rm datasets/testing_mask_dataset.zip
else
    echo "[*] Testing mask dataset already exists."
fi

# 5. Checkpoint management (Start vs. Continue training)
MODE=${1:-"continue"}
CHECKPOINT_DIR="PlacesTraining"

if [ "$MODE" == "start" ]; then
    if [ -d "$CHECKPOINT_DIR" ]; then
        BACKUP_DIR="${CHECKPOINT_DIR}_backup_$(date +%s)"
        echo "[*] Start mode selected. Renaming existing checkpoints directory to $BACKUP_DIR..."
        mv "$CHECKPOINT_DIR" "$BACKUP_DIR"
    fi
    mkdir -p "$CHECKPOINT_DIR"
    if [ -f "config.yml" ]; then
        cp "config.yml" "$CHECKPOINT_DIR/config.yml"
    elif [ -f "checkpoints_places/config.yml" ]; then
        cp "checkpoints_places/config.yml" "$CHECKPOINT_DIR/config.yml"
    else
        echo "[*] Downloading default config.yml from server..."
        curl -L -o "$CHECKPOINT_DIR/config.yml" https://files.lalithadithyan.dev/download/config.yml
    fi
else
    echo "[*] Continue mode selected (resuming from local checkpoints or server)."
    mkdir -p "$CHECKPOINT_DIR"
    
    python -c "
import os
import urllib.request
import json
import re

run_path = '$CHECKPOINT_DIR'
gen_dest = os.path.join(run_path, 'InpaintingModel_gen.pth')
dis_dest = os.path.join(run_path, 'InpaintingModel_dis.pth')

if os.path.exists(gen_dest) and os.path.exists(dis_dest):
    print('[*] Local checkpoints already exist in ' + run_path)
else:
    files_url = os.environ.get('FILES_SERVER_URL', 'https://files.lalithadithyan.dev')
    session = os.environ.get('C2_SESSION', 'DAVA')
    list_url = f'{files_url}/api/models?session={session}'
    print(f'[*] Fetching checkpoint listing from: {list_url}')
    try:
        req = urllib.request.Request(list_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        
        file_list = data.get('files', []) if isinstance(data, dict) else data
        
        gen_files = [(int(m.group(1)), f) for f in file_list
                     for m in [re.match(r'^DAVA_(\d{9})_InpaintingModel_gen\.pth$', f)] if m]
        dis_files = [(int(m.group(1)), f) for f in file_list
                     for m in [re.match(r'^DAVA_(\d{9})_InpaintingModel_dis\.pth$', f)] if m]
        
        if gen_files and dis_files:
            best_iter_gen, best_gen_name = max(gen_files, key=lambda x: x[0])
            best_iter_dis, best_dis_name = max(dis_files, key=lambda x: x[0])
            common_iter = min(best_iter_gen, best_iter_dis)
            
            best_gen_name = [x[1] for x in gen_files if x[0] == common_iter][0]
            best_dis_name = [x[1] for x in dis_files if x[0] == common_iter][0]
            
            print(f'[*] Downloading checkpoints at iteration {common_iter}...')
            urllib.request.urlretrieve(f'{files_url}/download/{best_gen_name}', gen_dest)
            urllib.request.urlretrieve(f'{files_url}/download/{best_dis_name}', dis_dest)
            print('[OK] Restored checkpoints from server.')
        else:
            print('[*] No checkpoints found on server for session ' + session + '. Starting fresh.')
    except Exception as e:
        print(f'[WARNING] Could not restore checkpoints from server: {e}')
"
    
    if [ ! -f "$CHECKPOINT_DIR/config.yml" ]; then
        if [ -f "config.yml" ]; then
            cp "config.yml" "$CHECKPOINT_DIR/config.yml"
        elif [ -f "checkpoints_places/config.yml" ]; then
            cp "checkpoints_places/config.yml" "$CHECKPOINT_DIR/config.yml"
        fi
    fi
fi

# 6. Dynamically update config settings for Places2
echo "[*] Auto-configuring config.yml settings for Places2..."
python -c "
import yaml
config_path = '$CHECKPOINT_DIR/config.yml'
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
cfg['TRAIN_INPAINT_IMAGE_FLIST'] = 'datasets/places365/places365_standard/train'
cfg['TEST_INPAINT_IMAGE_FLIST'] = 'datasets/places365/places365_standard/val'
cfg['TRAIN_MASK_FLIST'] = 'datasets/testing_mask_dataset'
cfg['TEST_MASK_FLIST'] = 'datasets/testing_mask_dataset'
cfg['FILTER_BY_SEG_MASK'] = True
with open(config_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"

# 7. Start the active training loop (supports DDP if multiple GPUs are available)
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
echo "[*] Auto-detecting GPU settings: found $NUM_GPUS GPU(s)..."

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "[*] Configured for Multi-GPU ($NUM_GPUS GPUs) DDP training."
    python -c "
import yaml
config_path = '$CHECKPOINT_DIR/config.yml'
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
cfg['GPU'] = list(range($NUM_GPUS))
with open(config_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"
    echo "================================================="
    echo "Starting PLACES2 SEM-Net DDP Training loop: [$C2_SESSION] on GPUs $(seq -s, 0 $(($NUM_GPUS-1)))"
    echo "================================================="
    torchrun --nproc_per_node=$NUM_GPUS --master_port=29501 main.py --model 2 --path "$CHECKPOINT_DIR"
else
    GPU_INDEX=0
    echo "[*] Configured for Single GPU ($GPU_INDEX) training."
    python -c "
import yaml
config_path = '$CHECKPOINT_DIR/config.yml'
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
cfg['GPU'] = [$GPU_INDEX]
with open(config_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"
    echo "================================================="
    echo "Starting PLACES2 SEM-Net Training loop: [$C2_SESSION] on GPU $GPU_INDEX"
    echo "================================================="
    python -u main.py --model 2 --path "$CHECKPOINT_DIR"
fi
