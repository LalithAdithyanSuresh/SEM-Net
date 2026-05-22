#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo "          SEM-Net Linux Server Setup Script               "
echo "=========================================================="

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

# Verify we are in the correct codebase directory
if [ ! -f "main.py" ]; then
    echo "ERROR: Could not find main.py. Make sure you are in the correct repository directory."
    exit 1
fi


# 2. Check for CUDA/NVCC compiler (needed for Mamba and DCNv3 CUDA kernels)
echo "Checking environment requirements..."
if command -v nvcc >/dev/null 2>&1; then
    echo "[OK] CUDA compiler (nvcc) found: $(nvcc --version | grep release)"
else
    echo "==========================================================="
    echo "WARNING: 'nvcc' (CUDA Compiler) was not found in your PATH."
    echo "Compiling causal-conv1d, mamba-ssm, and ops_dcnv3 will fail."
    echo "Please ensure CUDA Toolkit is installed and nvcc is in PATH."
    echo "==========================================================="
    read -p "Do you want to proceed anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
if command -v python3 >/dev/null 2>&1; then
    echo "[OK] Python 3 found: $(python3 --version)"
else
    echo "ERROR: python3 is not installed. Please install Python 3 first."
    exit 1
fi

# 3. Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment 'venv' already exists."
fi

# 4. Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade base packaging tools
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# 5. Install PyTorch first (crucial for compiling CUDA extensions during pip install)
echo "Installing PyTorch 2.0.1 and Torchvision 0.15.2..."
# We explicitly install PyTorch with CUDA 11.8 or default depending on system.
# Adjust index-url if needed, default PyTorch package contains CUDA runtimes.
pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

# 6. Install other requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
else
    echo "ERROR: requirements.txt not found. Cannot install dependencies."
    exit 1
fi

# 7. Download InternImage ops_dcnv3
if [ -d "src/ops_dcnv3" ] && [ -f "src/ops_dcnv3/make.sh" ]; then
    echo "src/ops_dcnv3 already exists and looks valid."
else
    echo "Downloading ops_dcnv3 from InternImage repository..."
    # Clean up dummy text file if it exists
    rm -f src/ops_dcnv3
    
    # Use sparse checkout to download only the ops_dcnv3 folder
    mkdir -p temp_internimage
    cd temp_internimage
    git init -q
    git remote add origin https://github.com/OpenGVLab/InternImage.git
    git config core.sparseCheckout true
    echo "classification/ops_dcnv3" >> .git/info/sparse-checkout
    git pull -q --depth 1 origin master
    cd ..
    
    # Move ops_dcnv3 to src/ and clean up temp files
    cp -r temp_internimage/classification/ops_dcnv3 src/
    rm -rf temp_internimage
    echo "ops_dcnv3 downloaded successfully."
fi

# 8. Compile ops_dcnv3
echo "Compiling ops_dcnv3 CUDA kernel..."
cd src/ops_dcnv3
# Fix execution permissions just in case
chmod +x make.sh
sh make.sh
cd ../..

echo "=========================================================="
echo "Setup completed successfully!"
echo "=========================================================="
echo "To activate this environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run training, update your config.yml with server-local dataset paths, and run:"
echo "  python train.py"
echo "=========================================================="
