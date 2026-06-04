#!/usr/bin/env bash
# setup.sh - Switch to ForMultiGPU branch and run setup.py
set -e

echo "=== Preparing SEM-Net Test Environment ==="

# 1. Switch to MultiGPU branch if not already on it
TARGET_BRANCH="ForMultiGPU"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")

if [ "$CURRENT_BRANCH" != "$TARGET_BRANCH" ]; then
    echo "Switching git branch to $TARGET_BRANCH..."
    git checkout "$TARGET_BRANCH"
else
    echo "Already on branch $TARGET_BRANCH."
fi

# 2. Find python executable (prefer venv if present)
PYTHON_EXEC="python3"
if [ -d "venv" ]; then
    if [ -f "venv/bin/python" ]; then
        PYTHON_EXEC="venv/bin/python"
    elif [ -f "venv/Scripts/python" ]; then
        # Windows Git Bash support
        PYTHON_EXEC="venv/Scripts/python"
    fi
fi

# 3. Launch setup.py
echo "Launching setup.py with: $PYTHON_EXEC"
exec $PYTHON_EXEC setup.py "$@"
