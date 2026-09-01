# Resolve the Python interpreter for every benchmark job.
#
# The job scripts call bare `python`. On this cluster that resolves to a system
# interpreter without numpy, so a prepared conda environment is put on PATH here
# instead of activating conda in each script. Override per job with BILLIGER_ENV
# (Magellan needs py_entitymatching, which only the `entitymatch` env has).
: "${BILLIGER_ENV:=/home/aasteine/miniconda3/envs/ditto-modern}"
export PATH="${BILLIGER_ENV}/bin:${PATH}"
if ! python -c "import numpy, pandas, sklearn" 2>/dev/null; then
    echo "[env] FATAL: ${BILLIGER_ENV} is missing numpy/pandas/sklearn" >&2
    exit 1
fi
echo "[env] python=$(command -v python) ($(python -c 'import sys;print(sys.version.split()[0])'))"
