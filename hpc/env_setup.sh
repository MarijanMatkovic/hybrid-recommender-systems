#!/bin/bash
# =====================================================================
# One-time Python-environment setup for Supek.
#
# Run this ONCE from the login node after cloning the repo:
#     bash hpc/env_setup.sh
#
# It creates a virtualenv under ~/diplomski_env and installs pinned
# dependencies. All subsequent PBS jobs just `source` it.
#
# On Supek the Python module is cray-python/3.11.7, which ships
# numpy/scipy linked against Cray LibSci (tuned BLAS/LAPACK). We create
# the venv with --system-site-packages so those optimised builds are
# inherited, and only pip-install what Cray doesn't provide.
#
# Check available versions with: module spider python
# Override the default like: PYTHON_MODULE=utils/python/3.12.2 bash hpc/env_setup.sh
# =====================================================================

set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-cray-python/3.11.7}"
ENV_DIR="${ENV_DIR:-$HOME/diplomski_env}"

echo ">>> Loading module: $PYTHON_MODULE"
module load "$PYTHON_MODULE"

if [ ! -d "$ENV_DIR" ]; then
    echo ">>> Creating virtualenv at $ENV_DIR (--system-site-packages)"
    python -m venv --system-site-packages "$ENV_DIR"
else
    echo ">>> Virtualenv already exists at $ENV_DIR (skipping create)"
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo ">>> Upgrading pip"
pip install --upgrade pip

# numpy and scipy come from cray-python -- don't reinstall them from
# PyPI, that would replace the LibSci-linked builds with plain wheels.
echo ">>> Installing project dependencies (on top of cray-python numpy/scipy)"
pip install --no-cache-dir \
    "pandas>=2.0,<4.0" \
    "scikit-learn>=1.3,<2.0" \
    "matplotlib>=3.7,<4.0" \
    "pytest>=8.0,<9.0"

echo ">>> Sanity check: which numpy/scipy are we using?"
python -c "import numpy, scipy; print('numpy:', numpy.__version__, numpy.__file__); print('scipy:', scipy.__version__, scipy.__file__)"

echo ">>> Running unit tests"
cd "$(dirname "$0")/.."
python -m pytest -q

echo ">>> Done. Activate later with:"
echo "    module load $PYTHON_MODULE"
echo "    source $ENV_DIR/bin/activate"
