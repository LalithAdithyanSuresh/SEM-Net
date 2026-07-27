#!/bin/bash
# =================================================================
# SEM-Net Automated CUDA, Mamba-SSM, DCNv3 & Training Launcher
# =================================================================

set -e

# 1. Ensure working directory is LAST/
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

LOG_FILE="setup_compilation.log"

# Stream all output live to BOTH the terminal AND setup_compilation.log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================================================="
echo "  SEM-Net Environment & Compilation Launcher    "
echo "================================================="
echo "Working directory: $(pwd)"
echo "Logging to: $LOG_FILE (Dual Live Terminal + File Output)"
echo "================================================="

# 2. Environment & CUDA Setup
USER_NAME=$(whoami)
export TORCH_HOME="/tmp/torch_cache_${USER_NAME}"
export MPLCONFIGDIR="/tmp/matplotlib_cache_${USER_NAME}"
export HF_HOME="/tmp/hf_cache_${USER_NAME}"
export PIP_CACHE_DIR="/tmp/pip_cache_${USER_NAME}"

mkdir -p "$TORCH_HOME" "$MPLCONFIGDIR" "$HF_HOME" "$PIP_CACHE_DIR" 2>/dev/null || true

if [ -d "/usr/local/cuda-12.4" ]; then
    export CUDA_HOME="/usr/local/cuda-12.4"
    export PATH="/usr/local/cuda-12.4/bin:${PATH}"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}"
fi
export MAX_JOBS=1

echo "[*] CUDA_HOME: ${CUDA_HOME:-'Not Set (Default CUDA)'}"
echo "[*] TORCH_HOME: $TORCH_HOME"

# 3. Activate Virtual Environment
if [ -d "../venv" ]; then
    echo "[*] Activating virtual environment (../venv)..."
    source ../venv/bin/activate
elif [ -d "venv" ]; then
    echo "[*] Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "[*] Activating virtual environment (.venv)..."
    source .venv/bin/activate
fi

# 4. Ensure PyTorch 2.1.2 (required for mamba-ssm 1.1.3 & CUDA 12)
echo "[*] Checking PyTorch version..."
if ! python -c "import torch; assert torch.__version__.startswith('2.1.2')" >/dev/null 2>&1; then
    echo "[*] Installing PyTorch 2.1.2 + torchvision 0.16.2 for mamba-ssm C++ compatibility..."
    pip install torch==2.1.2 torchvision==0.16.2 --extra-index-url https://download.pytorch.org/whl/cu121
fi

# 5. Check and Install Core Dependencies & PyTorch Boxing Patch
python -c "
import os, glob
paths = glob.glob(os.path.join('venv', 'lib', 'python*', 'site-packages', 'torch', 'include', 'ATen', 'core', 'boxing', 'impl', 'boxing.h')) + \
        glob.glob(os.path.join('../venv', 'lib', 'python*', 'site-packages', 'torch', 'include', 'ATen', 'core', 'boxing', 'impl', 'boxing.h'))
for path in paths:
    try:
        with open(path, 'r', encoding='utf-8') as f: content = f.read()
        start = '// has_ivalue_to<T> tests the presence/absence of instance method IValue::to<T>()'
        end = '// boxing predicates'
        if start in content and end in content:
            s_idx, e_idx = content.find(start), content.find(end)
            if 'struct has_ivalue_to : std::true_type {};' not in content[s_idx:e_idx]:
                block = '// has_ivalue_to<T> tests the presence/absence of instance method IValue::to<T>()\n//\ntemplate <class T, class Enable = void>\nstruct has_ivalue_to : std::true_type {};\n\n'
                with open(path, 'w', encoding='utf-8') as f: f.write(content[:s_idx] + block + content[e_idx:])
                print(f'[OK] Patched boxing.h in {path}')
    except Exception as e:
        print(f'Warning boxing patch: {e}')
" 2>/dev/null || true

# 6. Check & Compile causal-conv1d & mamba-ssm
echo "[*] Checking mamba-ssm installation..."
if ! python -c "import causal_conv1d; import mamba_ssm" >/dev/null 2>&1; then
    echo "[*] Compiling causal-conv1d and mamba-ssm with CUDA..."
    pip install wheel "setuptools<82" "numpy<2" packaging ninja
    pip install causal-conv1d==1.1.3.post1 mamba-ssm==1.1.3.post1 --no-build-isolation -v
else
    echo "[OK] causal-conv1d and mamba-ssm are installed and compiled."
fi

# 7. Check & Compile DCNv3 (ops_dcnv3)
echo "[*] Checking DCNv3 installation..."
if ! python -c "import torch; import DCNv3" >/dev/null 2>&1; then
    echo "[*] Setting up and compiling DCNv3 (src/ops_dcnv3)..."
    if [ ! -d "src/ops_dcnv3" ] || [ ! -f "src/ops_dcnv3/make.sh" ]; then
        if [ -d "../src/ops_dcnv3" ]; then
            mkdir -p src
            cp -r ../src/ops_dcnv3 src/
        fi
    fi

    if [ -d "src/ops_dcnv3" ] && [ -f "src/ops_dcnv3/make.sh" ]; then
        python -c "
import os
for root, _, files in os.walk('src/ops_dcnv3'):
    for f in files:
        if f.endswith(('.sh', '.py', '.cpp', '.cu', '.h')):
            p = os.path.join(root, f)
            try:
                data = open(p, 'rb').read().replace(b'\r\n', b'\n')
                open(p, 'wb').write(data)
            except Exception: pass
" 2>/dev/null || true

        cd src/ops_dcnv3
        chmod +x make.sh 2>/dev/null || true
        rm -rf build 2>/dev/null || true
        bash make.sh
        cd ../..
    fi
else
    echo "[OK] DCNv3 CUDA module is compiled and ready."
fi

# 8. Verify Checkpoint Config
CHECKPOINT_DIR="./checkpoints"
mkdir -p "$CHECKPOINT_DIR"
if [ -f "./config.yml" ]; then
    cp ./config.yml "$CHECKPOINT_DIR/config.yml"
fi

echo "================================================="
echo "[*] Environment setup & compilation completed!"
echo "[*] Launching Multi-GPU Training inside $(pwd)..."
echo "================================================="

torchrun --nproc_per_node=2 main.py --model 2 --path "$CHECKPOINT_DIR"
