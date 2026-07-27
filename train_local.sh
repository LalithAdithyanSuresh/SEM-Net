#!/bin/bash

# Move to the script's directory
cd "$(dirname "$0")"

echo "================================================="
echo "SEM-Net Local Training Script"
echo "================================================="

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "[*] Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "[*] Activating virtual environment (venv)..."
    source venv/bin/activate
else
    echo "[WARNING] No virtual environment found. Running with global Python."
fi

# Define path to checkpoint directory (which must contain config.yml)
CHECKPOINT_DIR=${1:-"./checkpoints"}

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "[*] Checkpoint directory '$CHECKPOINT_DIR' not found. Creating it..."
    mkdir -p "$CHECKPOINT_DIR"
fi

if [ ! -f "$CHECKPOINT_DIR/config.yml" ]; then
    echo "[ERROR] 'config.yml' not found in '$CHECKPOINT_DIR'."
    if [ -f "./config.yml" ]; then
        echo "[*] Copying template config.yml to '$CHECKPOINT_DIR'..."
        cp ./config.yml "$CHECKPOINT_DIR/"
    else
        echo "Please provide a valid config.yml"
        exit 1
    fi
fi

echo "[*] Starting local training using $CHECKPOINT_DIR..."

# Launch training locally on GPU 0 by default. It doesn't send logs to any remote server.
python -u main.py --model 2 --path "$CHECKPOINT_DIR"

if [ $? -eq 0 ]; then
    echo "[*] Training finished normally."
else
    echo "[!] Training process exited with an error."
fi
