#!/bin/bash

# Configuration
export C2_SERVER_URL="https://lalithadithyan.dev"
export FILES_SERVER_URL="https://files.lalithadithyan.dev"
export C2_SESSION="DAVA"

echo "====================================="
echo "Starting StrDiffusion via C2: [$C2_SESSION]"
echo "====================================="

# Navigate to the sibling StrDiffusion directory
cd ../StrDiffusion || { echo "StrDiffusion folder not found!"; exit 1; }

# Run the setup and evaluation script, answering default options automatically, and piping output to push_logs.py
echo -e "\n\n\n" | ./setup_and_evaluate_str_diffusion.sh 2>&1 | python -u ../SEM-Net/push_logs.py

echo "Evaluation Script Finished!"
