#!/bin/bash

# Configuration variables
export C2_SERVER_URL="https://lalithadithyan.dev"
export FILES_SERVER_URL="https://files.lalithadityan.dev"
export C2_SESSION="DAVA"

echo "================================================="
echo "LaMa Evaluation Shell Script Starting"
echo "================================================="

# Create datasets directory if not exists
mkdir -p datasets

# 1. Clone LaMa repo if not already cloned
if [ ! -d "lama" ]; then
    echo "[*] Cloning LaMa repository..."
    git clone https://github.com/advimman/lama.git
else
    echo "[*] LaMa repository already exists."
fi

# 2. Download and unzip model file if not already present
if [ ! -d "lama/big-lama" ]; then
    echo "[*] Downloading big-lama model weights..."
    curl -L -o big-lama.zip https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
    echo "[*] Unzipping model weights..."
    unzip big-lama.zip -d lama/
    rm big-lama.zip
else
    echo "[*] LaMa weights already exist."
fi

# 3. Download and unzip CelebA-HQ 256 test dataset if not already present
if [ ! -d "datasets/celeba_hq_256_test" ]; then
    echo "[*] Downloading CelebA-HQ 256 test dataset..."
    curl -L -o celeba_hq_256_test.zip https://files.lalithadityan.dev/download/celeba_hq_256_test.zip
    echo "[*] Unzipping CelebA-HQ 256 test dataset..."
    unzip celeba_hq_256_test.zip -d datasets/
    rm celeba_hq_256_test.zip
else
    echo "[*] CelebA-HQ 256 test dataset already exists."
fi

# 4. Install required packages
echo "[*] Installing dependencies..."
pip install "numpy<2.0.0" omegaconf webdataset pytorch-lightning kornia joblib hydra-core albumentations requests scikit-image

# 5. Run evaluation script
echo "[*] Starting LaMa evaluation..."
python -u evaluate_lama.py --model-path lama/big-lama --image-dir datasets/celeba_hq_256_test --mask-dir datasets/testing_mask_dataset --output-dir evaluation_results_lama

echo "================================================="
echo "LaMa Evaluation Shell Script Finished!"
echo "================================================="
