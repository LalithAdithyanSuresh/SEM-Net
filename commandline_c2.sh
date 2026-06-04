#!/usr/bin/env bash

# Live remote command loop – fetches commands from the C2 server (lalithadithyan.dev)
# Usage: ./commandline_c2.sh [session_name]
# The script runs indefinitely, executing any command returned by the server.
# Press Ctrl+C to stop.

# --------------------------------------------------------------------
# Configuration (same defaults as run_training_c2.sh)
export C2_SERVER_URL="https://lalithadithyan.dev"
export FILES_SERVER_URL="https://files.lalithadithyan.dev"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="./tmp/torch_cache"
export MPLCONFIGDIR="./tmp/matplotlib_cache"
export PIP_CACHE_DIR="./tmp/pip_cache"

# Session name handling
if [ -z "$1" ]; then
  export C2_SESSION="DAVA"
  echo "No session name provided. Using 'DAVA'."
else
  export C2_SESSION="$1"
  echo "Starting session: $C2_SESSION"
fi

# --------------------------------------------------------------------
# Main loop – fetch and execute remote shell commands
while true; do
  # 1. Retrieve a pending shell command (if any)
  python - <<'PY'
import os, requests, subprocess, json, sys
url = os.getenv('C2_SERVER_URL')
session = os.getenv('C2_SESSION')
try:
    resp = requests.get(f"{url}/api/pop_shell_command", params={"session": session}, timeout=5)
    if resp.status_code == 200:
        cmd = resp.json().get('shell_command')
        if cmd:
            print(f"[C2 REMOTE COMMAND] {cmd}")
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            out = (proc.stdout + '\n' + proc.stderr).strip()
            # Send output back to server
            log_url = f"{url}/api/logs"
            payload = {
                "lines": [f"[REMOTE OUTPUT] {l}" for l in out.split('\n')],
                "session": session
            }
            try:
                requests.post(log_url, json=payload, timeout=5)
            except Exception:
                pass
            print(out)
except Exception as e:
    # Silently ignore any network or execution errors
    pass
PY
  # Small pause to avoid hammering the server
  sleep 2
done
