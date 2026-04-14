#!/bin/bash
# =====================================================================
# Shared setup for every PBS job in this repo.
#
# Sourced at the top of each hpc/run_*.pbs script. Activates the Python
# environment, cd's to the project directory, ensures the results/ and
# logs/ folders exist, and prints a header with host, date, job id.
# =====================================================================

set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/diplomski_env}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"

module purge
module load "$PYTHON_MODULE"

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

# cd to the repository -- $PBS_O_WORKDIR is the directory we submitted
# the job from, which should be the repo root.
cd "${PBS_O_WORKDIR:-$PWD}"

mkdir -p results logs

echo "========================================================"
echo "  Job         : ${PBS_JOBNAME:-unknown} (${PBS_JOBID:-local})"
echo "  Host        : $(hostname)"
echo "  Date        : $(date -Is)"
echo "  Work dir    : $PWD"
echo "  Python      : $(python -V)"
echo "  Numpy       : $(python -c 'import numpy; print(numpy.__version__)')"
echo "  Scipy       : $(python -c 'import scipy; print(scipy.__version__)')"
echo "========================================================"
