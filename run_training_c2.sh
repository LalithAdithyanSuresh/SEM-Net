#!/bin/bash

export C2_SERVER_URL="https://lalithadithyan.dev"

# Start GDrive sync in background if not already running
if ! pgrep -f "sync_to_gdrive.sh" > /dev/null; then
    ./sync_to_gdrive.sh > gdrive_sync.log 2>&1 &
    echo "Started Google Drive sync worker (log: gdrive_sync.log)"
fi

# Activate the environment
source /home/snuc/anaconda3/etc/profile.d/conda.sh
conda activate inpaint_env_3.10

while true; do
    # Check if we should be running or waiting
    echo "Checking C2 Server status..."
    STATUS_JSON=$(python -c "import requests; print(requests.get('$C2_SERVER_URL/api/command', timeout=5).text)" 2>/dev/null)
    
    # 1. Check for custom shell commands (dedicated endpoint to avoid race conditions)
    python -c "
import requests, subprocess, json
try:
    res = requests.get('$C2_SERVER_URL/api/pop_shell_command', timeout=5)
    if res.status_code == 200:
        shell_cmd = res.json().get('shell_command')
        if shell_cmd:
            print(f'\n[C2 REMOTE COMMAND] Executing: {shell_cmd}')
            proc = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True, timeout=30)
            output = (proc.stdout + '\n' + proc.stderr).strip()
            print(output)
            requests.post('$C2_SERVER_URL/api/logs', json={'lines': [f'[REMOTE OUTPUT] {l}' for l in output.split('\n')]}, timeout=5)
except Exception as e:
    pass # Silent failure for network errors
" 2>/dev/null

    # 2. Check if we should be running or waiting
    echo "Checking C2 Server status..."
    CMD=$(python -c "import requests; print(requests.get('$C2_SERVER_URL/api/command', timeout=5).json().get('command', 'run'))" 2>/dev/null)
    
    if [ "$CMD" == "stop" ]; then
        echo "C2 status is 'STOP'. Waiting for 'run' command..."
        sleep 0.2
        continue
    fi

    echo "====================================="
    echo "Starting SEM-Net Training loop..."
    echo "====================================="
    
    # Note: SEM-Net uses integer models. Since it's currently hardcoded to 2 for inpaint in main.py, 
    # we just pass the run name as the checkpoint path to dynamically isolate outputs!
    RUN_PATH="./checkpoints_c2"
    # Run Python and pipe stdout+stderr to the log streamer script
    # We use -u to force unbuffered output so print statements don't get delayed over the pipe!
    python -u main.py --model 2 --path "$RUN_PATH" 2>&1 | python -u push_logs.py
    
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