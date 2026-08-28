#!/usr/bin/env bash
# Kiem tra model nao trong MODELS_TO_TEST (dinh nghia trong local_multi_model.py)
# da duoc `ollama pull` ve, in ra danh sach co san/con thieu, roi chay
# local_multi_model.py de so sanh risk-scoring giua cac model tren ca 3
# dataset va xuat ket qua ra cac file CSV.
#
# Usage (redirect log vao local_multi_model/ de gom chung voi CSV output):
#   ./run_local_multi_model.sh --purpose "mo ta muc dich lan chay" [--models ...] [--datasets ...] [--tag ...] [--runs N] \
#     2>&1 | tee local_multi_model/run_$(date +%Y%m%d_%H%M).log
# (--purpose la bat buoc, xem local_multi_model.py --help cho cac tuy chon con lai.
#  --runs N chay lai toan bo pipeline N lan doc lap, moi lan ra file CSV rieng
#  hau to _run1.._runN de so sanh do on dinh giua cac lan chay.)

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# May nay chay 2 Ollama instance: mac dinh (port 11434, model dung chung/khong
# lien quan) va instance rieng cua user (port 11435, noi cac model trong
# MODELS_TO_TEST duoc pull vao, khop voi OLLAMA_BASE_URL trong
# local_multi_model.py) - phai tro dung port 11435.
export OLLAMA_HOST="127.0.0.1:11435"

echo "=== Kiem tra Ollama (OLLAMA_HOST=$OLLAMA_HOST) ==="
if ! command -v ollama >/dev/null 2>&1; then
    echo "Loi: khong tim thay lenh 'ollama'. Cai Ollama truoc: https://ollama.com" >&2
    exit 1
fi

if ! ollama list >/dev/null 2>&1; then
    echo "Loi: khong ket noi duoc Ollama server tai $OLLAMA_HOST. Chay 'ollama serve' truoc." >&2
    exit 1
fi

echo "=== Doc danh sach model can test tu local_multi_model.py ==="
MODELS_TO_TEST="$(python3 -c "import local_multi_model as m; print('\n'.join(m.MODELS_TO_TEST))")"

if [ -z "$MODELS_TO_TEST" ]; then
    echo "Loi: khong doc duoc MODELS_TO_TEST tu local_multi_model.py." >&2
    exit 1
fi

PULLED_MODELS="$(ollama list | tail -n +2 | awk '{print $1}')"

AVAILABLE=()
MISSING=()

while IFS= read -r model; do
    [ -z "$model" ] && continue
    if echo "$PULLED_MODELS" | grep -qF "$model"; then
        AVAILABLE+=("$model")
        echo "  [OK]     $model"
    else
        MISSING+=("$model")
        echo "  [THIEU]  $model  (chay: ollama pull $model)"
    fi
done <<< "$MODELS_TO_TEST"

TOTAL=$(( ${#AVAILABLE[@]} + ${#MISSING[@]} ))
echo
echo "=== ${#AVAILABLE[@]}/${TOTAL} model da san sang ==="

if [ "${#AVAILABLE[@]}" -eq 0 ]; then
    echo "Loi: chua co model nao trong danh sach duoc pull. Dung lai, khong chay so sanh." >&2
    exit 1
fi

if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "(Model con thieu se bi local_multi_model.py tu dong bo qua khi chay: ${MISSING[*]})"
fi

echo
echo "=== Chay local_multi_model.py ==="
python3 local_multi_model.py "$@"

echo
echo "=== Hoan tat. File CSV da xuat (trong local_multi_model/) ==="
ls -la local_multi_model/local_multi_model_results*.csv local_multi_model/local_multi_model_dataset_timing*.csv local_multi_model/local_multi_model_summary*.csv 2>/dev/null || true
