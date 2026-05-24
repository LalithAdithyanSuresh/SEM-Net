#!/usr/bin/env bash

REPO_URL="https://github.com/LalithAdithyanSuresh/SEM-Net.git"
REPO_BRANCH="ForMultiGPU"
REPO_DIR="SEM-Net"
SCREEN_NAME="trainer"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 0: Re-launch inside a 'trainer' screen session if not already in one
# ──────────────────────────────────────────────────────────────────────────────
if [ -z "$STY" ]; then
    # Not inside a screen session — check if 'screen' is available
    if ! command -v screen &>/dev/null; then
        echo "ERROR: 'screen' is not installed. Install it with: sudo apt install screen"
        exit 1
    fi

    # Kill any stale 'trainer' session so we start fresh
    screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true

    echo "Launching inside screen session '$SCREEN_NAME'..."
    echo "  To reattach later: screen -r $SCREEN_NAME"
    echo ""

    # Re-exec this script inside a new detached screen, then immediately attach
    screen -dmS "$SCREEN_NAME" bash "$0" "$@"
    sleep 0.5
    exec screen -r "$SCREEN_NAME"
    # exec replaces the current shell — nothing below this runs outside the screen
fi

# ──────────────────────────────────────────────────────────────────────────────
# We are now running INSIDE the 'trainer' screen session
# ──────────────────────────────────────────────────────────────────────────────
set -e

echo "=== Running inside screen session: $STY ==="
echo ""

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
