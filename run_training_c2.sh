#!/bin/bash

export C2_SERVER_URL="https://lalithadithyan.dev"

while true; do
    echo "====================================="
    echo "Starting SEM-Net Training loop..."
    echo "====================================="
    
    # Inform C2 Server of local YML configs
    echo "Uploading available .yml models to C2..."
    python3 -c "import glob, requests; ymls=[x.split('/')[-1].split(r'\\')[-1] for x in glob.glob('./**/*.y*ml', recursive=True)]; requests.post('$C2_SERVER_URL/api/available_models', json={'models': ymls}, timeout=2) if ymls else None" 2>/dev/null
    
    # Fetch the model config name from C2 Server
    echo "Fetching active model from C2 Server..."
    MODEL_NAME=$(python3 -c "import requests; print(requests.get('$C2_SERVER_URL/api/command').json().get('model', 'sem_net.yaml'))" 2>/dev/null)
    
    if [ -z "$MODEL_NAME" ]; then
        MODEL_NAME="sem_net.yaml"
    fi
    
    echo "Active model selected: $MODEL_NAME"
    # Note: SEM-Net uses integer models. Since it's currently hardcoded to 2 for inpaint in main.py, 
    # we just pass the run name as the checkpoint path to dynamically isolate outputs!
    RUN_PATH="./checkpoints_${MODEL_NAME%.*}"

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
