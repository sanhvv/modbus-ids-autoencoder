# Moving Ollama + the pipeline to lab232-a04

## Step 0 — SSH into the new machine

```bash
ssh 23532764@lab232-a04.cs.curtin.edu.au
```

## Step 1 — Check whether the home directory is shared (NFS)

```bash
mount | grep " /home "
```

- If you see a line like `strdat01p:/linuxhome on /home type nfs` **identical to lab232-a01** -> the home directory is shared, everything below (Ollama binary, pulled models, repo code) is **already there**, skip Steps 2 and 3, go straight to Step 4.
- If it's different (not NFS, or a different mount point) -> do Steps 2 and 3 in full.

Quick check of whether files/models are already present:

```bash
ls -la ~/ollama/bin/ollama
ls ~/.ollama/models 2>/dev/null && echo "models present"
ls ~/local-llms/modbus-ids-autoencoder/local_multi_model_ae32_relu.py 2>/dev/null && echo "repo present"
```

If all 3 commands above succeed (no error) -> the home directory is indeed shared, **skip Steps 2 and 3**.

## Step 2 — Install Ollama (only if Step 1 confirms the home is NOT shared)

The lab machine has no `sudo` access, so install it portably (extract into home, no root needed) - the same approach already used on lab232-a01:

```bash
mkdir -p ~/ollama && cd ~/ollama

# Download the official release from ollama.com (check for the latest
# version at https://github.com/ollama/ollama/releases if you want a
# version other than 0.32.14)
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
tar -xzf ollama-linux-amd64.tgz

# Add to PATH + set a dedicated port (same as lab232-a01, to avoid the
# default port 11434 in case another shared instance is also running)
echo 'export PATH=$HOME/ollama/bin:$PATH' >> ~/.bashrc
echo 'export OLLAMA_HOST=127.0.0.1:11435' >> ~/.bashrc
source ~/.bashrc

ollama --version
```

## Step 3 — Pull the models (only if Step 1 confirms the home is NOT shared)

Start the server first:

```bash
cd ~/ollama
OLLAMA_HOST=127.0.0.1:11435 nohup ollama serve > ollama.log 2>&1 &
disown
sleep 3
OLLAMA_HOST=127.0.0.1:11435 ollama list   # check the server is running
```

Pull exactly the model list used in `local_multi_model_ae32_relu.py` (MODELS_TO_TEST), light -> heavy:

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

Storage note: ~44GB total for all 8 models (qwen3:14b ~9.3GB, gemma4:12b ~7.6GB, gemma4:e4b ~9.6GB, deepseek-r1:8b ~5.2GB, qwen3:8b ~5.2GB, qwen3:4b ~2.5GB, phi4-mini ~2.5GB, openthinker:7b ~4.7GB) - check free space before pulling all of them in one go:

```bash
df -h ~
```

## Step 4 — Start the Ollama server on a04 (always required, even if the home is shared)

The model/binary can be shared via NFS, but the **`ollama serve` process must run on each machine itself** (each machine has its own GPU). Run it from a fixed directory to avoid a broken-cwd bug (previously seen on lab232-a01 due to an NFS remount):

```bash
cd ~/ollama
OLLAMA_HOST=127.0.0.1:11435 nohup ollama serve > ollama.log 2>&1 &
disown
sleep 3
OLLAMA_HOST=127.0.0.1:11435 ollama list
```

If Step 1 confirmed a shared home, this `ollama list` command will already show all 8 models with no need to pull again.

## Step 5 — Check free GPU on a04 before running

```bash
nvidia-smi
```

Check `Memory-Usage` and the `Processes` section - make sure no other process (from another user on the shared machine) is holding most of the VRAM, as previously happened on lab232-a01.

## Step 6 — Run the pipeline

If the repo is shared via NFS, the code is already at `~/local-llms/modbus-ids-autoencoder/`, no need to clone again. Run as usual:

```bash
cd ~/local-llms/modbus-ids-autoencoder
./run_local_multi_model.sh --purpose "Running on lab232-a04 since a01 is busy" \
  2>&1 | tee local_multi_model_ae32_relu/run_$(date +%Y%m%d_%H%M)_a04.log
```

(or use `local_multi_model_ae32_relu_structured_prompt.py` to run the prompt_2 variant instead - see the README's "Prompt 2" section)

## Notes

- `run_local_multi_model.sh` already sets `OLLAMA_HOST=127.0.0.1:11435` internally, no need to export it manually when running through this script.
- If a Python/pip package (`openai`, `torch`, `pandas`, ...) isn't available on a04 because Python packages aren't shared across machines, reinstall it with: `python3 -m pip install --user openai` (the other packages are usually available via system modules - check with `python3 -c "import torch, pandas, sklearn"`).
