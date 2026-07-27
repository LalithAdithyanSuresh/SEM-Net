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
echo "Network remote control is disabled."
exit 0
