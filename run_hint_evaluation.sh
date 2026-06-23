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
    mkdir -p "$HINT_CHECKPOINT_DIR"
    
    # Download generator checkpoints if missing
    if [ ! -f "$HINT_CHECKPOINT_DIR/InpaintingModel_gen.pth" ]; then
        echo "[*] Downloading HINT CelebA-HQ generator weights from Google Drive..."
        gdown --id 1tsJ8vYuyX4vkQusuPRkDXtUqs3w4ddT4 -O "$HINT_CHECKPOINT_DIR/InpaintingModel_gen.pth"
    fi
    
    # Check if the generator file is corrupted HTML page and re-download
    if [ -f "$HINT_CHECKPOINT_DIR/InpaintingModel_gen.pth" ] && head -n 1 "$HINT_CHECKPOINT_DIR/InpaintingModel_gen.pth" | grep -q "^<"; then
        echo "[!] HINT generator weights appear to be a corrupted HTML file. Re-downloading..."
        rm -f "$HINT_CHECKPOINT_DIR/InpaintingModel_gen.pth"
        gdown --id 1tsJ8vYuyX4vkQusuPRkDXtUqs3w4ddT4 -O "$HINT_CHECKPOINT_DIR/InpaintingModel_gen.pth"
    fi

    # Download discriminator checkpoints if missing
    if [ ! -f "$HINT_CHECKPOINT_DIR/InpaintingModel_dis.pth" ]; then
        echo "[*] Downloading HINT CelebA-HQ discriminator weights from Google Drive..."
        gdown --id 162Xfx6XcqScGYLz8Cgjlq-WOLRjZrSFB -O "$HINT_CHECKPOINT_DIR/InpaintingModel_dis.pth"
    fi
    
    # Check if the discriminator file is corrupted HTML page and re-download
    if [ -f "$HINT_CHECKPOINT_DIR/InpaintingModel_dis.pth" ] && head -n 1 "$HINT_CHECKPOINT_DIR/InpaintingModel_dis.pth" | grep -q "^<"; then
        echo "[!] HINT discriminator weights appear to be a corrupted HTML file. Re-downloading..."
        rm -f "$HINT_CHECKPOINT_DIR/InpaintingModel_dis.pth"
        gdown --id 162Xfx6XcqScGYLz8Cgjlq-WOLRjZrSFB -O "$HINT_CHECKPOINT_DIR/InpaintingModel_dis.pth"
    fi

    # Copy config.yml if missing
    if [ ! -f "$HINT_CHECKPOINT_DIR/config.yml" ]; then
        echo "[*] config.yml not found in $HINT_CHECKPOINT_DIR. Resolving from HINT source repository..."
        if [ -f "HINT/checkpoints/config.yml" ]; then
            cp "HINT/checkpoints/config.yml" "$HINT_CHECKPOINT_DIR/config.yml"
            echo "[+] Successfully copied default config.yml from HINT/checkpoints/"
        elif [ -f "TEMP_QUALITATIVE/TEMP_QUALITATIVE/HINT/checkpoints/config.yml" ]; then
            cp "TEMP_QUALITATIVE/TEMP_QUALITATIVE/HINT/config.yml" "$HINT_CHECKPOINT_DIR/config.yml"
            echo "[+] Successfully copied default config.yml from TEMP_QUALITATIVE/TEMP_QUALITATIVE/HINT/checkpoints/"
        else
            echo "[WARNING] Default HINT config.yml not found. Creating a minimal config.yml..."
            echo -e "MODE: 2\nMODEL: 2\nBATCH_SIZE: 1\nINPUT_SIZE: 256\nGAN_LOSS: lsgan\nGPU: [0]" > "$HINT_CHECKPOINT_DIR/config.yml"
        fi
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
