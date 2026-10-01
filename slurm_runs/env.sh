# Resolve the Python interpreter for every benchmark job.
#
# The job scripts call bare `python`. Set BILLIGER_ENV to the prefix of a prepared
# conda environment (see environments/) to put it on PATH instead of activating
# conda in each script; otherwise the current `python` is used. Magellan needs
# py_entitymatching, which only the `entitymatch` environment has.
if [ -n "${BILLIGER_ENV:-}" ]; then
    export PATH="${BILLIGER_ENV}/bin:${PATH}"
fi
if ! python -c "import numpy, pandas, sklearn" 2>/dev/null; then
    echo "[env] FATAL: $(command -v python) is missing numpy/pandas/sklearn" >&2
    exit 1
fi
echo "[env] python=$(command -v python) ($(python -c 'import sys;print(sys.version.split()[0])'))"
