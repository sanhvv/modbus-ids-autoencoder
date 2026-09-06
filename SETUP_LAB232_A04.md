# Chuyển Ollama + pipeline sang lab232-a04

## Bước 0 — SSH sang máy mới

```bash
ssh 23532764@lab232-a04.cs.curtin.edu.au
```

## Bước 1 — Kiểm tra xem home directory có dùng chung (NFS) không

```bash
mount | grep " /home "
```

- Nếu thấy dòng `strdat01p:/linuxhome on /home type nfs` **giống hệt lab232-a01** → home directory dùng chung, mọi thứ dưới đây (Ollama binary, model đã pull, repo code) **đã có sẵn**, bỏ qua Bước 2 và 3, chuyển thẳng xuống Bước 4.
- Nếu khác (không phải NFS, hoặc mount point khác) → làm đầy đủ Bước 2, 3.

Kiểm tra nhanh xem file/model có sẵn chưa:

```bash
ls -la ~/ollama/bin/ollama
ls ~/.ollama/models 2>/dev/null && echo "models co san"
ls ~/local-llms/modbus-ids-autoencoder/local_multi_model_ae32_relu.py 2>/dev/null && echo "repo co san"
```

Nếu cả 3 lệnh trên đều ra kết quả (không lỗi) → home directory đúng là dùng chung, **bỏ qua Bước 2 và 3**.

## Bước 2 — Cài Ollama (chỉ làm nếu Bước 1 xác nhận KHÔNG dùng chung home)

Máy lab không có quyền `sudo`, nên cài kiểu portable (giải nén vào home, không cần root) — đúng cách đã dùng trên lab232-a01:

```bash
mkdir -p ~/ollama && cd ~/ollama

# Tai ban chinh thuc tu ollama.com (kiem tra version moi nhat tai
# https://github.com/ollama/ollama/releases neu muon dung ban khac 0.32.14)
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
tar -xzf ollama-linux-amd64.tgz

# Them vao PATH + set port rieng (giong lab232-a01, tranh dung port 11434
# mac dinh de khong dung voi instance dung chung khac neu co)
echo 'export PATH=$HOME/ollama/bin:$PATH' >> ~/.bashrc
echo 'export OLLAMA_HOST=127.0.0.1:11435' >> ~/.bashrc
source ~/.bashrc

ollama --version
```

## Bước 3 — Pull các model (chỉ làm nếu Bước 1 xác nhận KHÔNG dùng chung home)

Khởi động server trước:

```bash
cd ~/ollama
OLLAMA_HOST=127.0.0.1:11435 nohup ollama serve > ollama.log 2>&1 &
disown
sleep 3
OLLAMA_HOST=127.0.0.1:11435 ollama list   # kiem tra server da chay
```

Pull đúng danh sách model dùng trong `local_multi_model_ae32_relu.py` (MODELS_TO_TEST), theo thứ tự nhẹ → nặng:

```bash
export OLLAMA_HOST=127.0.0.1:11435
ollama pull phi4-mini
ollama pull qwen3:4b
ollama pull gemma4:e4b
ollama pull qwen3:8b
ollama pull openthinker:7b
ollama pull deepseek-r1:8b
ollama pull gemma4:12b
ollama pull qwen3:14b
```

Lưu ý dung lượng: tổng ~44GB cho cả 8 model (qwen3:14b ~9.3GB, gemma4:12b ~7.6GB, gemma4:e4b ~9.6GB, deepseek-r1:8b ~5.2GB, qwen3:8b ~5.2GB, qwen3:4b ~2.5GB, phi4-mini ~2.5GB, openthinker:7b ~4.7GB) — kiểm tra dung lượng trống trước khi pull hết 1 lượt:

```bash
df -h ~
```

## Bước 4 — Khởi động Ollama server trên a04 (luôn cần làm, kể cả khi dùng chung home)

Model/binary có thể dùng chung qua NFS, nhưng **tiến trình `ollama serve` phải tự chạy trên từng máy** (mỗi máy có GPU riêng). Chạy từ 1 thư mục cố định để tránh lỗi cwd bị hỏng (đã từng gặp trên lab232-a01 do NFS remount):

```bash
cd ~/ollama
OLLAMA_HOST=127.0.0.1:11435 nohup ollama serve > ollama.log 2>&1 &
disown
sleep 3
OLLAMA_HOST=127.0.0.1:11435 ollama list
```

Nếu Bước 1 xác nhận dùng chung home, lệnh `ollama list` này sẽ hiện luôn cả 8 model mà không cần pull lại.

## Bước 5 — Kiểm tra GPU trống trên a04 trước khi chạy

```bash
nvidia-smi
```

Xem `Memory-Usage` và mục `Processes` — đảm bảo không có process nào khác (của người dùng khác trên máy chung) đang chiếm phần lớn VRAM, giống tình huống đã gặp trên lab232-a01.

## Bước 6 — Chạy pipeline

Nếu repo dùng chung qua NFS, code đã có sẵn ở `~/local-llms/modbus-ids-autoencoder/`, không cần clone lại. Chạy như bình thường:

```bash
cd ~/local-llms/modbus-ids-autoencoder
./run_local_multi_model.sh --purpose "Chay tren lab232-a04 vi a01 dang busy" \
  2>&1 | tee local_multi_model_ae32_relu/run_$(date +%Y%m%d_%H%M)_a04.log
```

(hoặc dùng `local_multi_model_ae32_relu_structured_prompt.py` nếu muốn chạy bản prompt_2 — xem README mục "Prompt 2")

## Ghi chú

- `run_local_multi_model.sh` đã tự set `OLLAMA_HOST=127.0.0.1:11435` bên trong, không cần export lại thủ công khi chạy qua script này.
- Nếu Python/pip package (`openai`, `torch`, `pandas`, ...) chưa có trên a04 do không dùng chung home Python packages riêng biệt theo máy, cài lại bằng: `python3 -m pip install --user openai` (các package còn lại thường có sẵn qua module hệ thống, kiểm tra bằng `python3 -c "import torch, pandas, sklearn"`).
