#!/bin/bash

# Configuration variables
export C2_SERVER_URL="https://lalithadithyan.dev"
export FILES_SERVER_URL="https://files.lalithadithyan.dev"
export VALIDATION_SERVER_URL="https://validate.lalithadithyan.dev"
export C2_SESSION="DAVA"
export TORCH_HOME="./tmp/torch_cache"

echo "================================================="
echo "HINT Evaluation Shell Script Starting"
echo "================================================="

# Create datasets directory if not exists
mkdir -p datasets

# 0. Install required packages (ensures gdown is available for downloading weights/datasets)
echo "[*] Installing dependencies..."
pip install "numpy<2.0.0" omegaconf webdataset pytorch-lightning kornia joblib hydra-core requests scikit-image easydict opencv-python tabulate scikit-learn pyyaml pandas matplotlib packaging einops timm gdown

# 1. Clone HINT repo if not already present
if [ ! -d "HINT" ] && [ ! -d "TEMP_QUALITATIVE/TEMP_QUALITATIVE/HINT" ]; then
    echo "[*] Cloning HINT repository..."
    git clone https://github.com/ChrisChen1023/HINT.git
else
    echo "[*] HINT directory already exists."
fi

# 2. Check and resolve HINT checkpoints
HINT_CHECKPOINT_DIR="TEMP_QUALITATIVE/TEMP_QUALITATIVE/HINT/HINT_Validate_Places2"

if [ ! -d "$HINT_CHECKPOINT_DIR" ]; then
    HINT_CHECKPOINT_DIR="checkpoints/hint_celebahq"
    if [ ! -d "$HINT_CHECKPOINT_DIR" ] || [ ! -f "$HINT_CHECKPOINT_DIR/InpaintingModel_gen.pth" ] || [ ! -f "$HINT_CHECKPOINT_DIR/config.yml" ]; then
        echo "[*] Pre-trained weights not found at $HINT_CHECKPOINT_DIR. Downloading HINT CelebA-HQ checkpoint from Google Drive..."
        mkdir -p "$HINT_CHECKPOINT_DIR"
        # Download the files inside the folder using gdown folder mode
        gdown --folder https://drive.google.com/drive/folders/1DPmw5LSVxmRXoiLzPrIePXJHla0ek6E9 -O "$HINT_CHECKPOINT_DIR"
    else
        echo "[*] Using existing HINT CelebA-HQ checkpoints at $HINT_CHECKPOINT_DIR."
    fi
else
    echo "[*] Using existing Places2 checkpoints at $HINT_CHECKPOINT_DIR."
fi

echo "[*] HINT checkpoint path configured as: $HINT_CHECKPOINT_DIR"

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
# If you are using Places2 model, the script uses the default path: TEMP_QUALITATIVE/TEMP_QUALITATIVE/HINT/HINT_Validate_Places2
# If evaluating CelebA-HQ, pass your downloaded checkpoints path: --model-path checkpoints/hint_celebahq
echo "[*] Starting HINT evaluation..."
python -u evaluate_hint.py --model-path "$HINT_CHECKPOINT_DIR" --image-dir datasets/celeba_hq_256_test --mask-dir datasets/testing_mask_dataset --output-dir evaluation_results_hint --log-file hint_evaluation.log

echo "================================================="
echo "HINT Evaluation Shell Script Finished!"
echo "================================================="
