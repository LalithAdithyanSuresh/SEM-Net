#!/bin/bash

# Configuration variables
export C2_SERVER_URL="https://lalithadithyan.dev"
export FILES_SERVER_URL="https://files.lalithadithyan.dev"
export VALIDATION_SERVER_URL="https://validate.lalithadithyan.dev"
export C2_SESSION="DAVA"
export TORCH_HOME="./tmp/torch_cache"

echo "================================================="
echo "CMT Evaluation Shell Script Starting"
echo "================================================="

# Create datasets directory if not exists
mkdir -p datasets

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "[*] Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "[*] Activating virtual environment (venv)..."
    source venv/bin/activate
else
    echo "[WARNING] No virtual environment (.venv or venv) found. Running in system Python environment."
fi

# 0. Install required packages (ensures gdown and correct PyTorch version are available)
echo "[*] Installing dependencies..."
pip install torch==2.1.2 torchvision==0.16.2 --extra-index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0.0" omegaconf webdataset pytorch-lightning kornia joblib hydra-core requests scikit-image easydict opencv-python tabulate scikit-learn pyyaml pandas matplotlib packaging einops timm gdown

# 1. Clone CMT repo if not already cloned
if [ ! -d "CMT" ]; then
    echo "[*] Cloning CMT repository..."
    git clone https://github.com/keunsoo-ko/CMT.git
else
    echo "[*] CMT repository already exists."
fi

# 2. Download CelebA-HQ model weights if not already present
if [ ! -f "CMT/CelebA.pth" ]; then
    echo "[*] Downloading CMT CelebA model weights from Google Drive..."
    mkdir -p CMT
    gdown --id 1e6EbwGnMGgGXAn4QLffT_Zx_BbidBSbR -O CMT/CelebA.pth
else
    # Check if the file is an HTML page (starts with '<')
    if head -n 1 "CMT/CelebA.pth" | grep -q "^<"; then
        echo "[!] CMT CelebA model weights appear to be a corrupted HTML file. Re-downloading with gdown..."
        rm -f CMT/CelebA.pth
        gdown --id 1e6EbwGnMGgGXAn4QLffT_Zx_BbidBSbR -O CMT/CelebA.pth
    else
        echo "[*] CMT CelebA model weights already exist and are valid."
    fi
fi

# 3. Download and unzip CelebA-HQ 256 test dataset if not already present
if [ ! -d "datasets/celeba_hq_256_test" ]; then
    echo "[*] Downloading CelebA-HQ 256 test dataset..."
    curl -L -o celeba_hq_256_test.zip https://files.lalithadithyan.dev/download/celeba_hq_256_test.zip
    echo "[*] Unzipping CelebA-HQ 256 test dataset..."
    unzip celeba_hq_256_test.zip -d datasets/
    rm celeba_hq_256_test.zip
else
    echo "[*] CelebA-HQ 256 test dataset already exists."
fi

# 3.5 Download and unzip testing mask dataset if not already present
if [ ! -d "datasets/testing_mask_dataset" ]; then
    echo "[*] Downloading testing mask dataset..."
    curl -L -o testing_mask_dataset.zip https://files.lalithadithyan.dev/download/testing_mask_dataset.zip
    echo "[*] Unzipping testing mask dataset..."
    unzip testing_mask_dataset.zip -d datasets/
    rm testing_mask_dataset.zip
else
    echo "[*] Testing mask dataset already exists."
fi

# 5. Run evaluation script
echo "[*] Starting CMT evaluation..."
python -u evaluate_cmt.py --model-path CMT/CelebA.pth --image-dir datasets/celeba_hq_256_test --mask-dir datasets/testing_mask_dataset --output-dir evaluation_results_cmt --log-file cmt_evaluation.log

echo "================================================="
echo "CMT Evaluation Shell Script Finished!"
echo "================================================="
