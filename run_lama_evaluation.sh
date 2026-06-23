#!/bin/bash

# Configuration variables
export C2_SERVER_URL="https://lalithadithyan.dev"
export FILES_SERVER_URL="https://files.lalithadithyan.dev"
export C2_SESSION="DAVA"
export TORCH_HOME="./tmp/torch_cache"

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

# 2. Download and place CelebA-HQ model weights if not already present
if [ ! -d "lama/lama-regular-celebahq" ]; then
    echo "[*] Downloading LaMa CelebA-HQ model weights and config..."
    mkdir -p lama/lama-regular-celebahq/models
    curl -L -o lama/lama-regular-celebahq/config.yaml https://huggingface.co/camenduru/big-lama/resolve/main/lama-celeba-hq/lama-regular/config.yaml
    curl -L -o lama/lama-regular-celebahq/models/best.ckpt https://huggingface.co/camenduru/big-lama/resolve/main/lama-celeba-hq/lama-regular/models/best.ckpt
else
    echo "[*] LaMa CelebA-HQ model weights already exist."
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


# 4. Install required packages
echo "[*] Installing dependencies..."
pip install "numpy<2.0.0" omegaconf webdataset pytorch-lightning kornia joblib hydra-core "albumentations==0.5.2" requests scikit-image easydict opencv-python tabulate scikit-learn pyyaml pandas matplotlib packaging

# 5. Run evaluation script
echo "[*] Starting LaMa evaluation..."
python -u evaluate_lama.py --model-path lama/lama-regular-celebahq --image-dir datasets/celeba_hq_256_test --mask-dir datasets/testing_mask_dataset --output-dir evaluation_results_lama --log-file lama_evaluation.log

echo "================================================="
echo "LaMa Evaluation Shell Script Finished!"
echo "================================================="
