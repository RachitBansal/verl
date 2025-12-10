#!/bin/bash
#
# Validation script to check if the environment is set up correctly
# Run this before starting evaluation
#

set -e

echo "=========================================="
echo "Validating OLMo-2 Evaluation Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check counter
CHECKS_PASSED=0
CHECKS_FAILED=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((CHECKS_PASSED++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((CHECKS_FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# 1. Check if in verl directory
echo "1. Checking working directory..."
if [ -f "setup.py" ] && [ -d "verl" ]; then
    check_pass "In verl repository root"
else
    check_fail "Not in verl repository root. Please cd to /n/netscratch/dam_lab/Lab/brachit/verl"
fi

# 2. Check Python and verl installation
echo ""
echo "2. Checking Python and verl..."
if python3 -c "import verl" 2>/dev/null; then
    VERL_VERSION=$(python3 -c "import verl; print(getattr(verl, '__version__', 'unknown'))" 2>/dev/null)
    check_pass "verl package installed (version: ${VERL_VERSION})"
else
    check_fail "verl package not installed. Run: pip install -e ."
fi

# 3. Check required packages
echo ""
echo "3. Checking required packages..."
REQUIRED_PACKAGES=("torch" "transformers" "datasets" "pandas" "ray" "hydra")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import ${pkg}" 2>/dev/null; then
        check_pass "${pkg} installed"
    else
        check_fail "${pkg} not installed"
    fi
done

# 4. Check model checkpoint
echo ""
echo "4. Checking model checkpoint..."
MODEL_PATH="/n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/OLMo2-1B-stage1-50B/step14000-hf"

if [ -d "${MODEL_PATH}" ]; then
    check_pass "Model directory exists: ${MODEL_PATH}"
    
    # Check for required files
    if [ -f "${MODEL_PATH}/config.json" ]; then
        check_pass "config.json found"
    else
        check_fail "config.json not found in model directory"
    fi
    
    if [ -f "${MODEL_PATH}/tokenizer_config.json" ] || [ -f "${MODEL_PATH}/tokenizer.json" ]; then
        check_pass "Tokenizer files found"
    else
        check_warn "Tokenizer files not found (may cause issues)"
    fi
    
    # Check for model weights
    if [ -f "${MODEL_PATH}/pytorch_model.bin" ] || [ -f "${MODEL_PATH}/model.safetensors" ] || [ -f "${MODEL_PATH}/model.safetensors.index.json" ]; then
        check_pass "Model weights found"
    else
        check_fail "Model weights not found"
    fi
else
    check_fail "Model directory not found: ${MODEL_PATH}"
    echo "    Please update MODEL_PATH in evaluate_olmo2_math.sh"
fi

# 5. Check GPU availability
echo ""
echo "5. Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "${GPU_COUNT}" -gt 0 ]; then
        check_pass "${GPU_COUNT} GPU(s) detected"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | while read line; do
            echo "    ${line}"
        done
    else
        check_fail "No GPUs detected"
    fi
else
    check_fail "nvidia-smi not found. Are you on a GPU node?"
fi

# 6. Check data directories
echo ""
echo "6. Checking data directories..."
DATA_DIR="${HOME}/data"
if [ -d "${DATA_DIR}" ]; then
    check_pass "Data directory exists: ${DATA_DIR}"
else
    mkdir -p "${DATA_DIR}"
    check_pass "Created data directory: ${DATA_DIR}"
fi

# Check if datasets already exist
if [ -f "${DATA_DIR}/gsm8k/test.parquet" ]; then
    check_pass "GSM-8K dataset already prepared"
else
    check_warn "GSM-8K dataset not prepared (will be created on first run)"
fi

if [ -f "${DATA_DIR}/math/test.parquet" ]; then
    check_pass "MATH dataset already prepared"
else
    check_warn "MATH dataset not prepared (will be created on first run)"
fi

# 7. Check evaluation scripts
echo ""
echo "7. Checking evaluation scripts..."
SCRIPTS=(
    "evaluate_olmo2_math.sh"
    "scripts/create_fewshot_dataset.py"
    "scripts/evaluate_majority_voting.py"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "${script}" ]; then
        if [ -x "${script}" ]; then
            check_pass "${script} (executable)"
        else
            check_warn "${script} (not executable, run: chmod +x ${script})"
        fi
    else
        check_fail "${script} not found"
    fi
done

# 8. Check disk space
echo ""
echo "8. Checking disk space..."
HOME_SPACE=$(df -h "${HOME}" | tail -1 | awk '{print $4}')
echo "    Available space in HOME: ${HOME_SPACE}"
if df -h "${HOME}" | tail -1 | awk '{exit ($5+0 > 90)}'; then
    check_pass "Sufficient disk space"
else
    check_warn "Disk space low (>90% used)"
fi

# 9. Test data preprocessing (optional, quick test)
echo ""
echo "9. Testing data preprocessing (quick check)..."
if python3 -c "
import sys
sys.path.insert(0, 'examples/data_preprocess')
try:
    from gsm8k import extract_solution
    result = extract_solution('The answer is 5 + 3 = 8 #### 8')
    assert result == '8', f'Expected 8, got {result}'
    print('GSM-8K preprocessing functions work')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" 2>/dev/null; then
    check_pass "Data preprocessing functions working"
else
    check_fail "Data preprocessing test failed"
fi

# Summary
echo ""
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo -e "${GREEN}Passed: ${CHECKS_PASSED}${NC}"
if [ ${CHECKS_FAILED} -gt 0 ]; then
    echo -e "${RED}Failed: ${CHECKS_FAILED}${NC}"
fi
echo ""

if [ ${CHECKS_FAILED} -eq 0 ]; then
    echo -e "${GREEN}✓ Setup validated successfully!${NC}"
    echo ""
    echo "You can now run:"
    echo "  bash evaluate_olmo2_math.sh"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Setup validation failed. Please fix the issues above.${NC}"
    echo ""
    exit 1
fi


