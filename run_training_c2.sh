#!/bin/bash

# Configuration
export C2_SERVER_URL="https://lalithadithyan.dev"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Handle Dynamic Session Name
# Usage: ./run_training_c2.sh my_session_name
if [ -z "$1" ]; then
    export C2_SESSION="default"
    echo "No session name provided. Using 'default'."
else
    export C2_SESSION="$1"
    echo "Starting session: $C2_SESSION"
fi

# Start GDrive sync in background if not already running
if ! pgrep -f "sync_to_gdrive.sh" > /dev/null; then
    ./sync_to_gdrive.sh > gdrive_sync.log 2>&1 &
    echo "Started Google Drive sync worker (log: gdrive_sync.log)"
fi

# Environment already active (CAMINO_env)

while true; do
    echo "Checking C2 Server status for [$C2_SESSION]..."
    
    # 1. Check for custom shell commands
    python -c "
import requests, subprocess, os
try:
    url = os.environ.get('C2_SERVER_URL')
    sess = os.environ.get('C2_SESSION')
    res = requests.get(f'{url}/api/pop_shell_command', params={'session': sess}, timeout=5)
    if res.status_code == 200:
        shell_cmd = res.json().get('shell_command')
        if shell_cmd:
            print(f'\n[C2 REMOTE COMMAND] Executing: {shell_cmd}')
            proc = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True, timeout=30)
            output = (proc.stdout + '\n' + proc.stderr).strip()
            print(output)
            requests.post(f'{url}/api/logs', json={'lines': [f'[REMOTE OUTPUT] {l}' for l in output.split(\"\n\")], 'session': sess}, timeout=5)
except Exception: pass
" 2>/dev/null

    # 2. Check if we should be running or waiting
    CMD=$(python -c "import requests, os; url=os.environ.get('C2_SERVER_URL'); sess=os.environ.get('C2_SESSION'); print(requests.get(f'{url}/api/command', params={'session': sess}, timeout=5).json().get('command', 'run'))" 2>/dev/null)
    
    if [ "$CMD" == "stop" ]; then
        echo "C2 status is 'STOP'. Waiting for 'run' command..."
        sleep 5
        continue
    fi

    echo "====================================="
    echo "Starting SEM-Net Training loop: [$C2_SESSION]"
    echo "====================================="
    
    RUN_PATH="./PlacesTraining"
    
    # Run Python and pipe stdout+stderr to the log streamer script
    # Run with torchrun for DDP support (Multi-GPU)
    torchrun --nproc_per_node=2 --master_port=29501 main.py --model 2 --path "$RUN_PATH" 2>&1 | python -u push_logs.py
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 42 ]; then
        echo "Received Restart signal."
        git pull origin main
        sleep 2
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "Training session finished/stopped."
        sleep 5
    else
        echo "Training exited with error code $EXIT_CODE. Restarting in 10s..."
        sleep 10
    fi
done