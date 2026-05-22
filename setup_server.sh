#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

REPO_URL="https://github.com/LalithAdithyanSuresh/SEM-Net.git"
REPO_BRANCH="ForMultiGPU"
REPO_DIR="SEM-Net"

# 1. Clone the repository if not already inside a git repository
if [ ! -d ".git" ]; then
    if [ -d "$REPO_DIR" ]; then
        echo "Directory '$REPO_DIR' already exists. Entering it..."
        cd "$REPO_DIR"
    else
        echo "Cloning repository branch '$REPO_BRANCH' from $REPO_URL..."
        git clone -b "$REPO_BRANCH" "$REPO_URL"
        cd "$REPO_DIR"
    fi
else
    echo "Already inside a git repository. Continuing in current directory..."
fi

# 2. Verify setup_server.py exists in the current directory
if [ ! -f "setup_server.py" ]; then
    echo "ERROR: Could not find setup_server.py in $(pwd)"
    exit 1
fi

# 3. Hand off to the Python script
exec python3 setup_server.py "$@"
