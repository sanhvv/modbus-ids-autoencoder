#!/usr/bin/env bash
# Check which models in MODELS_TO_TEST (defined in local_multi_model_ae32_relu.py)
# have been `ollama pull`ed, print the available/missing list, then run
# local_multi_model_ae32_relu.py to compare risk-scoring across models on
# all 3 datasets and export results to CSV files.
#
# Usage (redirect the log into local_multi_model_ae32_relu/ to keep it with the CSV output):
#   ./run_local_multi_model.sh --purpose "run purpose description" [--models ...] [--datasets ...] [--tag ...] [--runs N] \
#     2>&1 | tee local_multi_model_ae32_relu/run_$(date +%Y%m%d_%H%M).log
# (--purpose is required, see local_multi_model_ae32_relu.py --help for the
#  other options. --runs N re-runs the entire pipeline N independent times,
#  each producing its own CSV files suffixed _run1.._runN to compare
#  consistency across runs.)

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# This machine runs 2 Ollama instances: the default one (port 11434,
# shared/unrelated models) and the user's own instance (port 11435, where
# the models in MODELS_TO_TEST are pulled to, matching OLLAMA_BASE_URL in
# local_multi_model_ae32_relu.py) - must point at port 11435.
export OLLAMA_HOST="127.0.0.1:11435"

echo "=== Checking Ollama (OLLAMA_HOST=$OLLAMA_HOST) ==="
if ! command -v ollama >/dev/null 2>&1; then
    echo "Error: 'ollama' command not found. Install Ollama first: https://ollama.com" >&2
    exit 1
fi

if ! ollama list >/dev/null 2>&1; then
    echo "Error: could not connect to the Ollama server at $OLLAMA_HOST. Run 'ollama serve' first." >&2
    exit 1
fi

echo "=== Reading the model list to test from local_multi_model_ae32_relu.py ==="
MODELS_TO_TEST="$(python3 -c "import local_multi_model_ae32_relu as m; print('\n'.join(m.MODELS_TO_TEST))")"

if [ -z "$MODELS_TO_TEST" ]; then
    echo "Error: could not read MODELS_TO_TEST from local_multi_model_ae32_relu.py." >&2
    exit 1
fi

PULLED_MODELS="$(ollama list | tail -n +2 | awk '{print $1}')"

AVAILABLE=()
MISSING=()

while IFS= read -r model; do
    [ -z "$model" ] && continue
    if echo "$PULLED_MODELS" | grep -qF "$model"; then
        AVAILABLE+=("$model")
        echo "  [OK]       $model"
    else
        MISSING+=("$model")
        echo "  [MISSING]  $model  (run: ollama pull $model)"
    fi
done <<< "$MODELS_TO_TEST"

TOTAL=$(( ${#AVAILABLE[@]} + ${#MISSING[@]} ))
echo
echo "=== ${#AVAILABLE[@]}/${TOTAL} models ready ==="

if [ "${#AVAILABLE[@]}" -eq 0 ]; then
    echo "Error: none of the listed models have been pulled. Stopping, not running the comparison." >&2
    exit 1
fi

if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "(Missing models will be automatically skipped by local_multi_model_ae32_relu.py: ${MISSING[*]})"
fi

echo
echo "=== Running local_multi_model_ae32_relu.py ==="
python3 local_multi_model_ae32_relu.py "$@"

echo
echo "=== Done. Exported CSV files (in local_multi_model_ae32_relu/) ==="
ls -la local_multi_model_ae32_relu/local_multi_model_results*.csv local_multi_model_ae32_relu/local_multi_model_dataset_timing*.csv local_multi_model_ae32_relu/local_multi_model_summary*.csv 2>/dev/null || true
