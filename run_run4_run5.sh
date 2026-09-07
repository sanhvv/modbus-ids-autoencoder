#!/usr/bin/env bash
# Chay tuan tu run4 roi run5 cua local_multi_model_ae32_relu_structured_prompt.py
# tren a01 (sau khi GPU a04 gap loi). Dung trong 1 screen session detached de
# song sot qua viec disconnect terminal.
set -uo pipefail

cd /home/23532764/local-llms/modbus-ids-autoencoder

echo "=== BAT DAU RUN4: $(date) ==="
python3 local_multi_model_ae32_relu_structured_prompt.py \
  --purpose "prompt2 run4 - chay lai tren a01 sau khi a04 gap loi GPU" \
  --tag run4 \
  2>&1 | tee "local_multi_model_ae32_relu_structured_prompt/run_$(date +%Y%m%d_%H%M)_run4.log"

echo "=== BAT DAU RUN5: $(date) ==="
python3 local_multi_model_ae32_relu_structured_prompt.py \
  --purpose "prompt2 run5 - lap lai lan 2 tren a01 de kiem chung do on dinh" \
  --tag run5 \
  2>&1 | tee "local_multi_model_ae32_relu_structured_prompt/run_$(date +%Y%m%d_%H%M)_run5.log"

echo "=== HOAN TAT CA RUN4 VA RUN5: $(date) ==="
