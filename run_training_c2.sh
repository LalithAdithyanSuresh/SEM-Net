#!/bin/bash

# Configuration
export C2_SERVER_URL="https://lalithadithyan.dev"
export FILES_SERVER_URL="https://files.lalithadithyan.dev"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="./tmp/torch_cache"
export MPLCONFIGDIR="./tmp/matplotlib_cache"
export PIP_CACHE_DIR="./tmp/pip_cache"


# Handle Dynamic Session Name
# Usage: ./run_training_c2.sh my_session_name
if [ -z "$1" ]; then
    export C2_SESSION="DAVA"
    echo "No session name provided. Using 'DAVA'."
else
    export C2_SESSION="$1"
    echo "Starting session: $C2_SESSION"
fi

# Start GDrive sync in background if not already running
if ! pgrep -f "sync_to_gdrive.sh" > /dev/null; then
    ./sync_to_gdrive.sh > gdrive_sync.log 2>&1 &
    echo "Started Google Drive sync worker (log: gdrive_sync.log)"
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated local virtual environment (venv)"
fi

echo "====================================="
echo "Starting Local SEM-Net Training loop: [$C2_SESSION]"
echo "====================================="

RUN_PATH="./PlacesTraining"

# Run Python training locally
torchrun --nproc_per_node=2 --master_port=29501 main.py --model 2 --path "$RUN_PATH"