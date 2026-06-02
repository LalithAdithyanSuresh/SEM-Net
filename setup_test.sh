#!/usr/bin/env bash

REPO_URL="https://github.com/LalithAdithyanSuresh/SEM-Net.git"
REPO_BRANCH="ForMultiGPU"
REPO_DIR="SEM-Net"
SCREEN_NAME="tester"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 0: Re-launch inside a 'tester' screen session if not already in one
# ──────────────────────────────────────────────────────────────────────────────
if [ -z "$STY" ]; then
    # Not inside a screen session — check if 'screen' is available
    if ! command -v screen &>/dev/null; then
        echo "ERROR: 'screen' is not installed. Install it with: sudo apt install screen"
        exit 1
    fi

    # Kill any stale 'tester' session so we start fresh
    screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true

    echo "Launching inside screen session '$SCREEN_NAME'..."
    echo "  To reattach later: screen -r $SCREEN_NAME"
    echo ""

    # Resolve absolute path of this script so screen can find it regardless of directory
    SCRIPT_PATH=$(realpath "$0" 2>/dev/null || readlink -f "$0" 2>/dev/null || echo "$0")

    # Re-exec this script inside a new detached screen, then immediately attach
    screen -dmS "$SCREEN_NAME" bash "$SCRIPT_PATH" "$@"
    sleep 0.5
    exec screen -r "$SCREEN_NAME"
    # exec replaces the current shell — nothing below this runs outside the screen
fi

# ──────────────────────────────────────────────────────────────────────────────
# We are now running INSIDE the 'tester' screen session
# ──────────────────────────────────────────────────────────────────────────────
set -e

echo "=== Running inside screen session: $STY ==="
echo ""

# 1. Clone or update the repository
if [ ! -d ".git" ]; then
    if [ -d "$REPO_DIR" ]; then
        echo "Directory '$REPO_DIR' already exists. Entering it..."
        cd "$REPO_DIR"
        echo "Attempting to pull latest changes from branch '$REPO_BRANCH'..."
        git fetch origin || echo "WARNING: git fetch failed."
        git checkout "$REPO_BRANCH" || echo "WARNING: git checkout failed."
        git pull origin "$REPO_BRANCH" || echo "WARNING: git pull failed (possibly due to local changes). Continuing with local files."
    else
        echo "Cloning repository branch '$REPO_BRANCH' from $REPO_URL..."
        git clone -b "$REPO_BRANCH" "$REPO_URL"
        cd "$REPO_DIR"
    fi
else
    echo "Already inside a git repository. Attempting to pull latest changes from branch '$REPO_BRANCH'..."
    git fetch origin || echo "WARNING: git fetch failed."
    git checkout "$REPO_BRANCH" || echo "WARNING: git checkout failed."
    git pull origin "$REPO_BRANCH" || echo "WARNING: git pull failed (possibly due to local changes). Continuing with local files."
fi

# 2. Verify setup_test.py exists in the current directory
if [ ! -f "setup_test.py" ]; then
    echo "ERROR: Could not find setup_test.py in $(pwd)"
    exit 1
fi

# 3. Hand off to the Python script
exec python3 setup_test.py "$@"
