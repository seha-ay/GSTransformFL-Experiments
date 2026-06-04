#!/bin/bash
# ============================================================
# setup.sh — Install dependencies for BrainMRI FL benchmark
# Run from repo root: bash setup.sh
# ============================================================
set -e
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
echo ""
echo "  BrainMRI FL Benchmark — Setup"
echo "  ────────────────────────────────────────"
echo ""
# ── Check Python ──────────────────────────────────────────────────────────────
echo "  [1/5] Checking Python..."
python3 -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'"
echo "  ✅  Python $(python3 -c 'import sys; v=sys.version_info; print(str(v.major)+"."+str(v.minor))')"
# ── Check CUDA and select CuPy variant ───────────────────────────────────────
echo ""
echo "  [2/5] Checking CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "  ❌  nvidia-smi not found — GPU required"
    exit 1
fi
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
else
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
fi
CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1)
if find /usr/local/cuda*/lib64 -name "libcufft.so.11*" 2>/dev/null | grep -q .; then
    CUPY_PKG="cupy-cuda12x"
elif find /usr/local/cuda*/targets/*/lib -name "libcufft.so.10*" 2>/dev/null | grep -q .; then
    CUPY_PKG="cupy-cuda11x"
else
    CUPY_PKG="cupy-cuda${CUDA_MAJOR}x"
fi
echo "  ✅  CUDA ${CUDA_VERSION} — using ${CUPY_PKG}"
# ── Install dependencies ──────────────────────────────────────────────────────
echo ""
echo "  [3/5] Installing dependencies..."
pip install --quiet \
    "numpy>=1.23.0" \
    "nvflare>=2.7.0" \
    "torch" \
    "torchvision" \
    "Pillow" \
    "scikit-learn" \
    "matplotlib" \
    "seaborn" \
    "tqdm" \
    "kaggle" \
    "packaging"
pip install --quiet ${CUPY_PKG}
echo "  ✅  Dependencies installed"
# ── Collect credentials interactively ────────────────────────────────────────
echo ""
echo "  [4/5] Credentials setup"
echo "  Credentials are used only for this session."
echo "  They are never written to any file or committed to git."
echo ""
if [ -z "${GITHUB_TOKEN}" ]; then
    read -s -p "  GitHub personal access token (repo scope): " GITHUB_TOKEN
    echo ""
fi
if [ -z "${KAGGLE_USERNAME}" ]; then
    read -p "  Kaggle username: " KAGGLE_USERNAME
fi
if [ -z "${KAGGLE_API_KEY}" ]; then
    read -s -p "  Kaggle API key: " KAGGLE_API_KEY
    echo ""
fi
export GITHUB_TOKEN
export KAGGLE_USERNAME
export KAGGLE_API_KEY
echo ""
echo "  ✅  Credentials collected for this session"
# ── Install GSTransformFL ─────────────────────────────────────────────────────
echo ""
echo "  [5/5] Installing GSTransformFL..."
pip install --quiet \
    "git+https://${GITHUB_TOKEN}@github.com/seha-ay/GSTransformFL.git"
echo "  ✅  GSTransformFL installed"
echo ""
echo "  ────────────────────────────────────────"
echo "  ✅  Setup complete."
echo ""
echo "  Next steps:"
echo "    1. source .env"
echo "    2. python data/convert.py"
echo "    3. Update SITE_DISTRIBUTIONS in config.py with actual class counts"
echo "    4. python data/split.py"
echo "  ────────────────────────────────────────"
echo ""