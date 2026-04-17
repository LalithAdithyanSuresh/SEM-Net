#!/bin/bash

export C2_SERVER_URL="https://lalithadithyan.dev"

while true; do
    echo "====================================="
    echo "Starting SEM-Net Training loop..."
    echo "====================================="
    
    # Note: SEM-Net uses integer models. Since it's currently hardcoded to 2 for inpaint in main.py, 
    # we just pass the run name as the checkpoint path to dynamically isolate outputs!
    RUN_PATH="./checkpoints_c2"

    # Run Python and pipe stdout+stderr to the log streamer script
    python3 main.py --model 2 --path "$RUN_PATH" 2>&1 | python3 push_logs.py
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 42 ]; then
        echo "====================================="
        echo "Received Restart & Pull signal (42)."
        echo "Executing git pull..."
        echo "====================================="
        git pull origin main
        echo "Restarting trainer..."
        sleep 2
    else
        echo "Training exited normally or crashed with code $EXIT_CODE."
        echo "Waiting 10 seconds before automated restart. Press Ctrl+C to abort."
        sleep 10
    fi
done
