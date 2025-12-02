#!/usr/bin/env bash
# setup_verl_uv.sh - Setup verl environment from scratch using uv
# 
# This script will:
# 1. Check if uv is installed
# 2. Create a new virtual environment
# 3. Install PyTorch with CUDA support
# 4. Install verl and all dependencies
# 5. Install optional dependencies (vllm, flash-attn, etc.)

set -e  # Exit on any error

echo "=================================================="
echo "Setting up verl environment with uv"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Environment configuration
ENV_NAME="verl-uv"
PROJECT_ROOT="/n/netscratch/dam_lab/Lab/brachit/verl"
CACHE_ROOT="/n/netscratch/dam_lab/Lab/brachit/.cache"
PYTHON_VERSION="3.10"

# CUDA configuration - adjust if needed
CUDA_VERSION="12.4"
TORCH_VERSION="2.5.1"

# Set cache directories to avoid filling up home directory
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TMPDIR="${CACHE_ROOT}/tmp"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"

# Create cache directories
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${TMPDIR}" "${HF_HOME}" "${TORCH_HOME}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Environment name: $ENV_NAME"
echo "  Python version: $PYTHON_VERSION"
echo "  CUDA version: $CUDA_VERSION"
echo "  PyTorch version: $TORCH_VERSION"
echo "  Project root: $PROJECT_ROOT"
echo "  Cache root: $CACHE_ROOT"
echo ""
echo "All caches and temporary files will be stored in: $CACHE_ROOT"
echo "This avoids filling up your home directory!"
echo ""

# Step 1: Check if uv is installed
echo -e "${YELLOW}[1/6] Checking for uv...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${RED}ERROR: uv is not installed.${NC}"
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "or:"
    echo "  pip install uv"
    exit 1
fi
echo -e "${GREEN}✓ uv is installed: $(uv --version)${NC}"
echo ""

# Step 2: Create virtual environment
echo -e "${YELLOW}[2/6] Creating virtual environment with Python ${PYTHON_VERSION}...${NC}"
cd "$PROJECT_ROOT"

# Remove old environment if it exists
if [ -d ".venv-${ENV_NAME}" ]; then
    echo "Removing old environment..."
    rm -rf ".venv-${ENV_NAME}"
fi

# Create new environment
uv venv ".venv-${ENV_NAME}" --python "${PYTHON_VERSION}"
echo -e "${GREEN}✓ Virtual environment created at .venv-${ENV_NAME}${NC}"
echo ""

# Activate the environment
source ".venv-${ENV_NAME}/bin/activate"

# Step 3: Install PyTorch with CUDA support
echo -e "${YELLOW}[3/6] Installing PyTorch ${TORCH_VERSION} with CUDA ${CUDA_VERSION}...${NC}"
# Using uv pip to install PyTorch from the official PyTorch index
uv pip install torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Step 4: Install core dependencies
echo -e "${YELLOW}[4/6] Installing core dependencies...${NC}"
uv pip install \
    accelerate \
    codetiming \
    datasets \
    dill \
    hydra-core \
    "numpy<2.0.0" \
    pandas \
    peft \
    "pyarrow>=19.0.0" \
    pybind11 \
    pylatexenc \
    "ray[default]>=2.41.0" \
    torchdata \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    transformers \
    wandb \
    "packaging>=20.0" \
    tensorboard \
    uvicorn \
    fastapi \
    latex2sympy2_extended
echo -e "${GREEN}✓ Core dependencies installed${NC}"
echo ""

# Step 5: Install GPU-specific dependencies
echo -e "${YELLOW}[5/6] Installing GPU-specific dependencies...${NC}"

# Load CUDA module if available (required for compiling flash-attn)
echo "Loading CUDA module..."
if command -v module &> /dev/null; then
    module load cuda/12.4.1-fasrc01 2>/dev/null || module load cuda/12.4 2>/dev/null || echo "Warning: Could not load CUDA module"
    echo "CUDA loaded: $(which nvcc 2>/dev/null || echo 'nvcc not found')"
else
    echo "Warning: module command not available, attempting to continue..."
fi

# Install flash-attn (this can take a while)
echo "Installing flash-attn (this may take several minutes)..."
echo "Note: This requires CUDA toolkit and may take 5-10 minutes to compile..."

# Try to install flash-attn with proper environment
if uv pip install flash-attn --no-build-isolation; then
    echo -e "${GREEN}✓ flash-attn installed successfully${NC}"
else
    echo -e "${YELLOW}⚠ Warning: flash-attn installation failed${NC}"
    echo "flash-attn is optional but recommended for performance."
    echo "You can try installing it manually later with:"
    echo "  module load cuda/12.4.1-fasrc01"
    echo "  source .venv-verl-uv/bin/activate"
    echo "  uv pip install flash-attn --no-build-isolation"
    echo ""
    echo "Continuing with installation..."
fi

# Install liger-kernel
echo "Installing liger-kernel..."
if uv pip install liger-kernel; then
    echo -e "${GREEN}✓ liger-kernel installed${NC}"
else
    echo -e "${YELLOW}⚠ Warning: liger-kernel installation failed (optional)${NC}"
fi

# Install vllm
echo "Installing vllm..."
if uv pip install "vllm>=0.8.5,<=0.11.0"; then
    echo -e "${GREEN}✓ vllm installed${NC}"
else
    echo -e "${RED}✗ ERROR: vllm installation failed${NC}"
    echo "vllm is required for verl. Please check the error messages above."
    exit 1
fi

# Install math verification
echo "Installing math-verify..."
if uv pip install math-verify; then
    echo -e "${GREEN}✓ math-verify installed${NC}"
else
    echo -e "${YELLOW}⚠ Warning: math-verify installation failed (optional)${NC}"
fi

echo -e "${GREEN}✓ GPU-specific dependencies installation complete${NC}"
echo ""

# Step 6: Install verl in editable mode
echo -e "${YELLOW}[6/6] Installing verl package in editable mode...${NC}"
cd "$PROJECT_ROOT"
uv pip install -e .
echo -e "${GREEN}✓ verl package installed${NC}"
echo ""

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
echo ""

# Check Python version
echo "Python version:"
python --version
echo ""

# Check PyTorch and CUDA
echo "PyTorch version and CUDA availability:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"
echo ""

# Check verl
echo "verl version:"
python -c "import verl; print(f'verl installed successfully')" 2>/dev/null && echo -e "${GREEN}✓ verl import successful${NC}" || echo -e "${RED}✗ verl import failed${NC}"
echo ""

# Check other key packages
echo "Checking key dependencies:"
python -c "import vllm; print(f'vllm: {vllm.__version__}')" 2>/dev/null && echo -e "${GREEN}✓ vllm${NC}" || echo -e "${YELLOW}⚠ vllm not available${NC}"
python -c "import flash_attn; print('flash-attn: installed')" 2>/dev/null && echo -e "${GREEN}✓ flash-attn${NC}" || echo -e "${YELLOW}⚠ flash-attn not available${NC}"
python -c "import ray; print(f'ray: {ray.__version__}')" 2>/dev/null && echo -e "${GREEN}✓ ray${NC}" || echo -e "${YELLOW}⚠ ray not available${NC}"
echo ""

echo "=================================================="
echo -e "${GREEN}Setup complete!${NC}"
echo "=================================================="
echo ""
echo "To activate this environment in the future, run:"
echo -e "${YELLOW}  source ${PROJECT_ROOT}/.venv-${ENV_NAME}/bin/activate${NC}"
echo ""
echo "Or add this to your SLURM script:"
echo -e "${YELLOW}  source ${PROJECT_ROOT}/.venv-${ENV_NAME}/bin/activate${NC}"
echo ""
echo "To deactivate:"
echo -e "${YELLOW}  deactivate${NC}"
echo ""

