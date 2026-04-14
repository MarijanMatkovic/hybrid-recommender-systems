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
# If your project uses a different Python version, change PYTHON_MODULE
# below. Check available versions with:   module avail python
# =====================================================================

set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"
ENV_DIR="${ENV_DIR:-$HOME/diplomski_env}"

echo ">>> Loading module: $PYTHON_MODULE"
module load "$PYTHON_MODULE"

if [ ! -d "$ENV_DIR" ]; then
    echo ">>> Creating virtualenv at $ENV_DIR"
    python -m venv "$ENV_DIR"
else
    echo ">>> Virtualenv already exists at $ENV_DIR (skipping create)"
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo ">>> Upgrading pip"
pip install --upgrade pip

echo ">>> Installing dependencies"
pip install --no-cache-dir \
    "numpy>=1.26,<3.0" \
    "scipy>=1.11,<2.0" \
    "pandas>=2.0,<4.0" \
    "scikit-learn>=1.3,<2.0" \
    "matplotlib>=3.7,<4.0" \
    "pytest>=8.0,<9.0"

echo ">>> Running unit tests as a sanity check"
cd "$(dirname "$0")/.."
python -m pytest -q

echo ">>> Done. Activate later with:"
echo "    source $ENV_DIR/bin/activate"
