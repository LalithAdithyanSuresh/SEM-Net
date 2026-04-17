#!/bin/bash

export C2_SERVER_URL="https://lalithadithyan.dev"

# Activate the environment
source /home/snuc/anaconda3/etc/profile.d/conda.sh
conda activate inpaint_env_3.10

while true; do
    # Check if we should be running or waiting
    echo "Checking C2 Server status..."
    CMD=$(python -c "import requests; print(requests.get('$C2_SERVER_URL/api/command', timeout=5).json().get('command', 'run'))" 2>/dev/null)
    
    if [ "$CMD" == "stop" ]; then
        echo "C2 status is 'STOP'. Waiting for 'run' command..."
        sleep 10
        continue
    fi

    echo "====================================="
    echo "Starting SEM-Net Training loop..."
    echo "====================================="
    
    # Note: SEM-Net uses integer models. Since it's currently hardcoded to 2 for inpaint in main.py, 
    # we just pass the run name as the checkpoint path to dynamically isolate outputs!
    RUN_PATH="./checkpoints_c2"

    # Run Python and pipe stdout+stderr to the log streamer script
    python main.py --model 2 --path "$RUN_PATH" 2>&1 | python push_logs.py
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 42 ]; then
        echo "====================================="
        echo "Received Restart & Pull signal (42)."
        echo "Executing git pull..."
        echo "====================================="
        git pull origin main
        echo "Restarting trainer..."
        sleep 2
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "====================================="
        echo "Training session finished/stopped."
        echo "Restarting loop to wait for next command..."
        echo "====================================="
        sleep 2
    else
        echo "Training exited with error code $EXIT_CODE."
        echo "Waiting 10 seconds before automated restart. Press Ctrl+C to abort."
        sleep 10
    fi
done