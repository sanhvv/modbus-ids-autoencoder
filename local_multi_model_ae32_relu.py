"""
Compare risk-scoring across several local LLMs (via Ollama) on ALL 3
ICS-SimLab datasets (Intelligent Electronic Device, Smart Grid, Water
Bottle Factory).

This is the standalone version, extended from "local llms.py" (which only
ran 1 dataset and had to run inside a notebook that already had the
required variables/functions) - this script loads + cleans the datasets
itself, loads the pretrained 32-dim autoencoder and its saved threshold for
all 3 datasets, then calls the local models for risk-scoring, so it can be
run standalone with:
    python local_multi_model_ae32_relu.py --purpose "..." 2>&1 | tee local_multi_model_ae32_relu/run_$(date +%Y%m%d_%H%M).log
(output CSVs also go into local_multi_model_ae32_relu/, created right at
import time so a shell redirect into that folder works from the very first run)

MODELS BEING TESTED (already pulled, running on a GTX 3060 GPU):
    phi4-mini, qwen3:4b, gemma4:e4b, qwen3:8b, openthinker:7b, deepseek-r1:8b,
    gemma4:12b, qwen3:14b

REQUIREMENTS BEFORE RUNNING:
1. Ollama installed and running (`ollama serve`).
2. The models to test already pulled (the script automatically skips any
   model that hasn't been pulled, see check_model_available()).
3. The 32-dim autoencoder's model/threshold files already present in the repo:
     <dataset>_ae_model.pt, <dataset>_threshold.txt
   (e.g. smart_grid_ae_model.pt, smart_grid_threshold.txt)

SPEED WARNING: number of model calls = n_models x n_datasets x n_attack_types x
REPEATS_PER_PROMPT. Default REPEATS_PER_PROMPT=1 for a quick trial run;
increase to 2-3 to measure the consistency of the risk score.
"""

import re
import os
import sys
import time
import argparse
import statistics
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from openai import OpenAI

from retrain_ae_9dim import DATA_DIR, DATASET_FILENAMES, find_dataset_csv

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Model list ordered LIGHT -> HEAVY. Runs on a GTX 3060 GPU.
MODELS_TO_TEST = [
    "phi4-mini",
    "qwen3:4b",
    "gemma4:e4b",
    "qwen3:8b",
    "openthinker:7b",
    "deepseek-r1:8b",
    "gemma4:12b",
    "qwen3:14b",
]

DATASETS = list(DATASET_FILENAMES.keys())

# Attack type -> display name (same as the "Complete Pipeline" cell in the notebook)
ATTACKS = {
    1: "address scan",
    2: "function code scan",
    3: "device identification attack",
    4: "naive sensor read",
    5: "sporadic sensor measurement injection",
    6: "force listen mode",
    7: "restart communication",
    8: "data flood attack",
}

REPEATS_PER_PROMPT = 1      # increase to measure risk score consistency
REQUEST_TIMEOUT_SEC = 600   # 12B/14B models on a GTX 3060 can still take tens of seconds
# Ollama defaults to a 4096-token context window if num_ctx isn't set.
# "Thinking" models (gemma4:12b, qwen3, deepseek-r1, ...) generate a
# <think>...</think> block before answering and can burn through the whole
# 4096-token budget on that thinking, getting cut off before it can produce
# the actual answer -> response.choices[0].message.content ends up empty,
# no error, no risk score (see local_multi_model_results_1.csv, the
# gemma4:12b column: completion_tokens ~3800 but output/risk_score empty on
# all 3 datasets). Increase num_ctx so the model has room for both thinking
# and the answer.
NUM_CTX = 16384
# This machine runs 2 Ollama instances: the default one (11434, shared/
# unrelated models) and the user's own instance (11435, where the models in
# MODELS_TO_TEST are pulled to) - must point at port 11435, not the default.
OLLAMA_BASE_URL = "http://localhost:11435/v1"
# Used by check_model_available() (lists models via the OpenAI-compat API,
# unaffected by the num_ctx bug below). use_llm_local() calls the native
# endpoint directly below (no "/v1") so options.num_ctx is actually applied.
OLLAMA_NATIVE_BASE_URL = OLLAMA_BASE_URL.removesuffix("/v1")

OUTPUT_DETAIL_CSV = "local_multi_model_results.csv"
OUTPUT_DATASET_TIMING_CSV = "local_multi_model_dataset_timing.csv"
OUTPUT_SUMMARY_CSV = "local_multi_model_summary.csv"

# This script's output CSVs all go here (see the same note in
# retrain_ae_9dim.py - created right at import time so a shell redirect
# into this folder works from the very first run).
OUTPUT_DIR = Path("local_multi_model_ae32_relu")
OUTPUT_DIR.mkdir(exist_ok=True)


local_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",       # Ollama doesn't check the key, but the SDK requires a value
    timeout=REQUEST_TIMEOUT_SEC,
)


# ============================================================
# 2. 32-DIM AUTOENCODER + PREPROCESSING PIPELINE (from the "Complete Pipeline" cell)
# ============================================================

# CLASS:    AutoEncoder
# PURPOSE:  32-dim latent space autoencoder, fixed nn.ReLU() activation (not
#           tuned) - the ORIGINAL architecture from ics_simlab_sanh.ipynb
#           (cell "Autoencoder (AE)"), as opposed to the tuned/experimental
#           variants in retrain_ae_9dim.py (9-dim), tune_ae_9dim.py (Linear
#           + several activations), tune_ae_lstm_vae.py (LSTM/VAE). Must
#           match 1:1 the architecture used when training *_ae_model.pt
#           (loaded via load_ae_model()) - changing the architecture here
#           without retraining would corrupt the state_dict.
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def clean_dataset_dl(df, multiclass=True):
    df_c = df.copy()

    if "protocol" in df_c.columns:
        df_c = df_c[df_c['protocol'] == 'MODBUS']
    df_c = df_c.drop("protocol", axis=1)

    df_c = df_c.drop("time", axis=1)

    mean_rtt = df_c["tcp_analysis_ack_rtt"].mean()
    df_c["tcp_analysis_ack_rtt"] = df_c["tcp_analysis_ack_rtt"].fillna(mean_rtt)

    df_c = df_c.fillna(0)
    df_c = df_c.replace("N/A", 0)

    df_c['modbus_data'] = df_c['modbus_data'].astype(str).apply(lambda x: x[:12])

    df_c['ip_id'] = df_c['ip_id'].apply(lambda x: int(str(x), 16))
    df_c['ip_checksum'] = df_c['ip_checksum'].apply(lambda x: int(str(x), 16))
    df_c['tcp_flags'] = df_c['tcp_flags'].apply(lambda x: int(str(x), 16))
    df_c['modbus_data'] = df_c['modbus_data'].apply(
        lambda x: int(str(x).replace("0x", ""), 16) if str(x).replace("0x", "") else None
    )

    df_c = df_c.drop("ip_src", axis=1)
    df_c = df_c.drop("ip_dst", axis=1)
    df_c = df_c.drop("ether_src_mac", axis=1)
    df_c = df_c.drop("ether_dst_mac", axis=1)

    df_c = df_c.drop("ip_checksum", axis=1)
    df_c = df_c.drop("modbus_data", axis=1)

    std_scaler = StandardScaler()
    dontStand = ["attack_binary", "attack_obj", "attack_specific",
                 "modbus_func_code",
                 "ip_flags_df", "ip_flags_mf",
                 "orig_index"]

    standardized = df_c.drop(columns=dontStand)
    ignore = df_c[dontStand]
    features_scaled = pd.DataFrame(
        std_scaler.fit_transform(standardized),
        columns=standardized.columns,
        index=standardized.index)
    df_c = pd.concat([features_scaled, ignore], axis=1)

    if "protocol" in df_c.columns:
        valid_protocols = ["TCP", "UDP", "ICMP", "ARP", "DNS", "HTTP", "HTTPS", "FTP", "SSH"]
        df_c = df_c[df_c["protocol"].isin(valid_protocols)]
        df_c = pd.get_dummies(df_c, columns=['protocol'])

    df_c['attack_binary'] = df_c['attack_binary'].astype(int)
    df_c['attack_obj'] = df_c['attack_obj'].astype(int)
    df_c['attack_specific'] = df_c['attack_specific'].astype(int)

    df_c = df_c.drop("tcp_stream", axis=1)
    df_c = df_c.drop("frame_time_relative", axis=1)

    if multiclass:
        df_c = df_c.drop("attack_binary", axis=1)
        df_c = df_c.drop("attack_obj", axis=1)
    else:
        df_c = df_c.drop("attack_specific", axis=1)
        df_c = df_c.drop("attack_obj", axis=1)

    df_attack8 = df_c[df_c["attack_specific"] == 8]
    df_others = df_c[df_c["attack_specific"] != 8]
    df_attack8_reduced = df_attack8.sample(frac=0.10, random_state=42)
    df_c = pd.concat([df_others, df_attack8_reduced])

    return df_c


def process_dl_dataset(df_orig):
    df_orig_with_indicies = df_orig.copy()
    df_orig_with_indicies["orig_index"] = df_orig.index

    df_orig_with_indicies_clean = clean_dataset_dl(df_orig_with_indicies)

    if "is_attack" not in df_orig_with_indicies_clean.columns:
        df_orig_with_indicies_clean["is_attack"] = (df_orig_with_indicies_clean["attack_specific"] != 0).astype(int)
        if "attack_specific" in df_orig_with_indicies_clean.columns:
            df_orig_with_indicies_clean.drop("attack_specific", axis=1, inplace=True)

    df_normal = df_orig_with_indicies_clean[df_orig_with_indicies_clean["is_attack"] == 0]
    df_attack = df_orig_with_indicies_clean[df_orig_with_indicies_clean["is_attack"] == 1]

    target_size = int(len(df_orig_with_indicies_clean) * 0.5 / 2)

    df_normal_down = df_normal.sample(n=min(len(df_normal), target_size), random_state=42)
    df_attack_down = df_attack.sample(n=min(len(df_attack), target_size), random_state=42)

    df_orig_with_indicies_sampled = pd.concat([df_normal_down, df_attack_down]).sample(frac=1, random_state=42).reset_index(drop=True)

    X = df_orig_with_indicies_sampled.drop(columns=["is_attack", "orig_index"])
    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    inference_indices = X.index

    return X_tensor, inference_indices, df_orig_with_indicies, df_orig_with_indicies_sampled


def load_ae_model(X_tensor, file_model_name):
    input_dim = X_tensor.shape[1]
    autoencoder = AutoEncoder(input_dim)
    autoencoder.load_state_dict(torch.load(file_model_name))
    autoencoder.eval()
    return autoencoder


def inference_ae_model(autoencoder, X_tensor, inference_indices, df_orig_with_indicies, df_orig_with_indicies_sampled, threshold):
    with torch.no_grad():
        reconstructed = autoencoder(X_tensor)
        reconstructed_mse = torch.mean(torch.pow(X_tensor - reconstructed, 2), dim=1).numpy()

    anomaly_labels = (reconstructed_mse > threshold).astype(int)

    anomaly_indices = inference_indices[anomaly_labels == 1]
    cleaned_anomalies = df_orig_with_indicies_sampled.loc[anomaly_indices]

    cleaned_anomalies_indices = cleaned_anomalies["orig_index"]

    original_anomalies = df_orig_with_indicies[df_orig_with_indicies["orig_index"].isin(cleaned_anomalies_indices)]
    original_anomalies = original_anomalies.sort_values(by="orig_index")
    return original_anomalies


def select_anomalous_packet(original_anomalies, df_orig, attack_specific):
    middle = len(original_anomalies[original_anomalies["attack_specific"] == attack_specific]) // 2
    attack_row = original_anomalies[original_anomalies["attack_specific"] == attack_specific].iloc[middle]
    attack_row_index = attack_row["orig_index"]
    original_packet = df_orig.iloc[attack_row_index]
    return original_packet


def extract_packet_info(original_packet, df_orig):
    orig_packet_info = {
        "ip_src": original_packet["ip_src"],
        "ip_dst": original_packet["ip_dst"],
        "protocol": original_packet["protocol"],
        "ip_len": original_packet["ip_len"],
        "tcp_analysis_ack_rtt": original_packet["tcp_analysis_ack_rtt"],
        "tcp_analysis_bytes_in_flight": original_packet["tcp_analysis_bytes_in_flight"],
        "frame_time_delta": original_packet["frame_time_delta"],
        "modbus_function_code": original_packet["modbus_func_code"],
        "modbus_data": original_packet["modbus_data"],
    }

    k = 4
    frame_time_relative = original_packet["frame_time_relative"]

    df_numeric_time = df_orig.copy()
    df_numeric_time["frame_time_relative"] = pd.to_numeric(df_numeric_time["frame_time_relative"], errors='coerce')

    last_k_mask = (df_numeric_time["frame_time_relative"] >= frame_time_relative - k) & (df_numeric_time["frame_time_relative"] <= frame_time_relative)
    last_k_orig_packets = df_numeric_time[last_k_mask]

    total_packet_duration = df_numeric_time["frame_time_relative"].max() - df_numeric_time["frame_time_relative"].min()
    normal_packet_rate = round(len(df_numeric_time[df_numeric_time["attack_specific"].isna()]) / total_packet_duration, 2)

    packet_rate = len(last_k_orig_packets) / k

    address_columns = ["ether_src_mac", "ether_dst_mac", "ip_src", "ip_dst"]
    most_common_addresses = {}
    for col in address_columns:
        most_common_addresses[col] = last_k_orig_packets[col].mode()[0]

    orig_flow_info = {
        "packet_rate": packet_rate,
        "average_packet_rate": normal_packet_rate,
        "most_common_ip_src": most_common_addresses["ip_src"],
        "most_common_ip_dst": most_common_addresses["ip_dst"],
        "most_common_mac_src": most_common_addresses["ether_src_mac"],
        "most_common_mac_dst": most_common_addresses["ether_dst_mac"],
    }
    return orig_packet_info, orig_flow_info


def create_prompt(orig_packet_info, orig_flow_info):
    prompt = f'''
  You are an intrusion detection system tasked with monitoring an Industrial
  Control System network. The following information has been extract from a
  potentially anomalous network packet.

  IP/TCP packet information:
  IP Source Address: {orig_packet_info["ip_src"]}
  IP Destination Address: {orig_packet_info["ip_dst"]}
  Protocol: {orig_packet_info["protocol"]}
  IP Length: {orig_packet_info["ip_len"]}
  Frame Time Delta: {orig_packet_info["frame_time_delta"]}
  '''

    if pd.notna(orig_packet_info["tcp_analysis_ack_rtt"]):
        prompt += f'''
  TCP Analysis Ack RTT: {orig_packet_info["tcp_analysis_ack_rtt"]}
  TCP Analysis Bytes In Flight: {orig_packet_info["tcp_analysis_bytes_in_flight"]}
  '''

    if orig_packet_info["protocol"] == "MODBUS":
        prompt += f'''
  Modbus packet information.
  Function Code: {orig_packet_info["modbus_function_code"]}
  Modbus PDU: {orig_packet_info["modbus_data"]}
  '''

    prompt += f'''
  Network flow information:
  Packet flow rate: {orig_flow_info["packet_rate"]} packets per second

  An example of anomalous packet flow would have a much higher flow rate than {orig_flow_info["average_packet_rate"]} packets per second.

  Analyse the network behaviour. Answer within 250 characters. You cannot exceed this limit.

  Determine a risk score out of 10 to represent the
  severity of this anomalous activity. It must formatted in the last sentence as
  Risk Score: X/10
  '''
    return prompt


def prepare_dataset(dataset_name):
    """Load the raw csv, run the already-trained 32-dim autoencoder, and
    return (df_orig, original_anomalies) for the risk-scoring step."""
    csv_path = find_dataset_csv(DATASET_FILENAMES[dataset_name])
    df_orig = pd.read_csv(csv_path)

    X_tensor, inference_indices, df_orig_with_indicies, df_orig_with_indicies_sampled = process_dl_dataset(df_orig)

    file_prefix = dataset_name.lower().replace(" ", "_")
    model_file = file_prefix + "_ae_model.pt"
    threshold_file = file_prefix + "_threshold.txt"

    autoencoder = load_ae_model(X_tensor, model_file)
    with open(threshold_file) as f:
        threshold = float(f.read().strip())

    original_anomalies = inference_ae_model(
        autoencoder, X_tensor, inference_indices,
        df_orig_with_indicies, df_orig_with_indicies_sampled, threshold,
    )

    return df_orig, original_anomalies


# ============================================================
# 3. CALLING THE LOCAL MODEL (from "local llms.py")
# ============================================================

def check_model_available(client, model_name: str) -> bool:
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        return any(model_name in m for m in available)
    except Exception as e:
        print(f"  [WARNING] Could not connect to Ollama server: {e}")
        return False


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def use_llm_local(client, prompt: str, model_name: str):
    # IMPORTANT: call the Ollama native /api/chat endpoint directly via
    # requests, NOT the OpenAI-compat client (client.chat.completions.create)
    # here. Manually verified (curl + `ollama ps`) that the OpenAI-compat
    # /v1/chat/completions endpoint on the Ollama build running here
    # (0.32.14) SILENTLY IGNORES "options": {"num_ctx": ...} (both nested
    # under "options" and flattened as top-level "num_ctx") - the model is
    # always reloaded with the default num_ctx=4096 no matter what the
    # client sends (`ollama ps` still reports context_length=4096). The
    # native /api/chat endpoint applies it correctly (`ollama ps` reports
    # the correct context_length + size_vram increases accordingly). This
    # is exactly why the first fix attempt (using extra_body via the
    # OpenAI-compat client) had no effect: gemma4:12b was still cut off at
    # exactly 4096 tokens (prompt+completion) same as before the fix.
    #
    # "think": False - the README used to say "tried think:false, no way to
    # turn it off" but that was tried via the OpenAI-compat path (affected
    # by the same bug above, the option silently dropped). Via the native
    # endpoint, think:false works correctly: the model answers directly
    # without generating a <think> block/"thinking" field, much faster
    # (e.g. gemma4:12b: ~2-5s instead of hundreds of seconds or empty output).
    response = requests.post(
        f"{OLLAMA_NATIVE_BASE_URL}/api/chat",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.2, "num_ctx": NUM_CTX},
        },
        timeout=REQUEST_TIMEOUT_SEC,
    )
    if not response.ok:
        # response.raise_for_status() only reports the status line (e.g.
        # "400 Client Error: Bad Request for url: ..."), WITHOUT the body -
        # but the body is where Ollama writes the real reason (e.g. out of
        # VRAM, model runner crash...). Attach the body to the message so
        # it can be debugged immediately next time instead of guessing.
        raise RuntimeError(
            f"{response.status_code} {response.reason} for url: {response.url} "
            f"- response body: {response.text[:1000]}"
        )
    data = response.json()
    output = data.get("message", {}).get("content")
    usage = _Usage(data.get("prompt_eval_count"), data.get("eval_count"))
    return output, usage


def strip_think_block(text: str) -> str:
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_risk_score(text: str):
    if not text:
        return None
    text = strip_think_block(text)
    match = re.search(r"Risk Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def check_format_compliance(text: str, max_chars: int = 250):
    text = strip_think_block(text)
    has_risk_score = extract_risk_score(text) is not None
    body = re.split(r"Risk Score:", text, flags=re.IGNORECASE)[0] if text else ""
    within_length = len(body.strip()) <= max_chars
    return has_risk_score, within_length


# ============================================================
# 4. MAIN LOOP: dataset -> model -> attack type -> N repeats
# ============================================================

def build_prompts_for_dataset(df_orig, original_anomalies, attacks):
    """Build one prompt per attack type up front (shared across all models/
    repeats to ensure a fair comparison on the same selected packet)."""
    prompts = {}
    for attack_specific, attack_name in attacks.items():
        if not (original_anomalies["attack_specific"] == attack_specific).any():
            print(f"  [SKIPPED] No anomaly detected by the AE for attack '{attack_name}' on this dataset.")
            continue
        original_packet = select_anomalous_packet(original_anomalies, df_orig, attack_specific)
        orig_packet_info, orig_flow_info = extract_packet_info(original_packet, df_orig)
        prompts[attack_specific] = (attack_name, create_prompt(orig_packet_info, orig_flow_info))
    return prompts


def run_comparison(models_to_test, datasets, attacks, run_purpose=""):
    results = []
    dataset_timing = []

    for dataset_name in datasets:
        print(f"\n{'#'*60}")
        print(f"DATASET: {dataset_name}")
        print(f"{'#'*60}")

        df_orig, original_anomalies = prepare_dataset(dataset_name)
        prompts = build_prompts_for_dataset(df_orig, original_anomalies, attacks)

        for model_name in models_to_test:
            print(f"\n{'='*60}")
            print(f"MODEL: {model_name}  (dataset: {dataset_name})")
            print(f"{'='*60}")

            if not check_model_available(local_client, model_name):
                print(f"  [SKIPPED] Model '{model_name}' not ready on Ollama. "
                      f"Run: ollama pull {model_name}")
                continue

            dataset_model_start = time.time()

            for attack_specific, (attack_name, prompt) in prompts.items():
                for repeat_idx in range(REPEATS_PER_PROMPT):
                    print(f"  [{attack_name}] attempt {repeat_idx + 1}/{REPEATS_PER_PROMPT}...",
                          end=" ", flush=True)

                    start_t = time.time()
                    try:
                        output, usage = use_llm_local(local_client, prompt, model_name)
                        latency = time.time() - start_t
                        error = None
                    except Exception as e:
                        output = None
                        usage = None
                        latency = time.time() - start_t
                        error = str(e)
                        print(f"ERROR: {error}")

                    risk_score = extract_risk_score(output) if output else None
                    has_risk_score, within_length = (
                        check_format_compliance(output) if output else (False, False)
                    )

                    results.append({
                        "run_purpose": run_purpose,
                        "dataset": dataset_name,
                        "model": model_name,
                        "attack_type": attack_name,
                        "attack_specific": attack_specific,
                        "repeat": repeat_idx + 1,
                        "prompt": prompt,
                        "output": output,
                        "risk_score": risk_score,
                        "format_ok_has_score": has_risk_score,
                        "format_ok_length": within_length,
                        "latency_sec": round(latency, 2),
                        "prompt_tokens": usage.prompt_tokens if usage else None,
                        "completion_tokens": usage.completion_tokens if usage else None,
                        "error": error,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    })

                    if output:
                        print(f"OK ({latency:.1f}s, risk={risk_score})")

            dataset_model_time = time.time() - dataset_model_start
            n_calls = len(prompts) * REPEATS_PER_PROMPT
            dataset_timing.append({
                "model": model_name,
                "dataset": dataset_name,
                "total_time_sec": round(dataset_model_time, 2),
                "n_calls": n_calls,
            })
            print(f"  -> Time to run model '{model_name}' on dataset '{dataset_name}': "
                  f"{dataset_model_time:.2f}s ({n_calls} calls)")

    return pd.DataFrame(results), pd.DataFrame(dataset_timing)


# ============================================================
# 5. FINAL RESULTS SUMMARY (all 3 datasets combined, per model)
# ============================================================

def summarize_results(df_results: pd.DataFrame, df_timing: pd.DataFrame) -> pd.DataFrame:
    if df_results.empty:
        print("No results (no model may have been ready on Ollama).")
        return pd.DataFrame()

    summary_rows = []

    for model_name, group in df_results.groupby("model"):
        valid = group[group["error"].isna()]
        n_total = len(group)
        n_success = len(valid)

        format_compliance = (
            (valid["format_ok_has_score"] & valid["format_ok_length"]).mean() * 100
            if n_success > 0 else 0
        )

        avg_latency = valid["latency_sec"].mean() if n_success > 0 else None
        p95_latency = (
            valid["latency_sec"].quantile(0.95) if n_success > 0 else None
        )

        consistency_scores = []
        for (dataset_name, attack_type), attack_group in valid.groupby(["dataset", "attack_type"]):
            scores = attack_group["risk_score"].dropna().tolist()
            if len(scores) > 1:
                consistency_scores.append(statistics.pstdev(scores))
        avg_std_risk_score = (
            statistics.mean(consistency_scores) if consistency_scores else None
        )

        avg_completion_tokens = (
            valid["completion_tokens"].mean() if n_success > 0 else None
        )

        total_time_sec = df_timing[df_timing["model"] == model_name]["total_time_sec"].sum()

        summary_rows.append({
            "model": model_name,
            "success_rate_%": round(n_success / n_total * 100, 1) if n_total else 0,
            "format_compliance_%": round(format_compliance, 1),
            "avg_latency_sec": round(avg_latency, 2) if avg_latency is not None else None,
            "p95_latency_sec": round(p95_latency, 2) if p95_latency is not None else None,
            "avg_risk_score_stddev": (
                round(avg_std_risk_score, 2) if avg_std_risk_score is not None else None
            ),
            "avg_completion_tokens": (
                round(avg_completion_tokens, 1) if avg_completion_tokens is not None else None
            ),
            "total_time_sec": round(total_time_sec, 2),
        })

    return pd.DataFrame(summary_rows).sort_values("avg_latency_sec")


# ============================================================
# 6. RUN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare risk-scoring across several local LLMs on the ICS-SimLab dataset.")
    parser.add_argument(
        "--purpose", required=True,
        help="Required: a short description of this run's purpose (e.g. 'rerun "
             "gemma4:12b after fixing num_ctx'), printed to the log and stored "
             "in the run_purpose CSV column, for later aggregation of experiments.")
    parser.add_argument(
        "--models", default=None,
        help=f"Comma-separated list of models to run (default: all "
             f"{len(MODELS_TO_TEST)} models). E.g.: --models gemma4:12b")
    parser.add_argument(
        "--datasets", default=None,
        help=f"Comma-separated list of datasets to run (default: all 3 "
             f"datasets). E.g.: --datasets 'Smart Grid,Water Bottle Factory'")
    parser.add_argument(
        "--tag", default=None,
        help="Suffix added to the output CSV filenames (e.g. 'gemma4_ctxfix') "
             "so it doesn't overwrite a previous full run's results. Leave "
             "blank to use the default filenames.")
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of times to re-run the ENTIRE pipeline (each run calls "
             "the LLM again from scratch, not repeating the same answer) to "
             "compare consistency across independent runs. Each run writes "
             "its own CSV files (suffixed _run1, _run2, ...). Default 1 "
             "(single run, filenames unchanged, no suffix added).")
    args = parser.parse_args()
    if args.runs < 1:
        sys.exit("Error: --runs must be >= 1")
    return args


if __name__ == "__main__":
    args = parse_args()

    models_to_run = MODELS_TO_TEST
    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in requested if m not in MODELS_TO_TEST]
        if unknown:
            sys.exit(f"Error: model(s) not found in MODELS_TO_TEST: {unknown}")
        models_to_run = requested

    datasets_to_run = DATASETS
    if args.datasets:
        requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
        unknown = [d for d in requested if d not in DATASETS]
        if unknown:
            sys.exit(f"Error: dataset(s) not found in DATASETS: {unknown}")
        datasets_to_run = requested

    base_suffix = f"_{args.tag}" if args.tag else ""

    for run_idx in range(1, args.runs + 1):
        run_suffix = base_suffix + (f"_run{run_idx}" if args.runs > 1 else "")
        output_detail_csv = OUTPUT_DIR / OUTPUT_DETAIL_CSV.replace(".csv", f"{run_suffix}.csv")
        output_timing_csv = OUTPUT_DIR / OUTPUT_DATASET_TIMING_CSV.replace(".csv", f"{run_suffix}.csv")
        output_summary_csv = OUTPUT_DIR / OUTPUT_SUMMARY_CSV.replace(".csv", f"{run_suffix}.csv")

        if args.runs > 1:
            print(f"\n{'#'*60}\nRUN {run_idx}/{args.runs}\n{'#'*60}")
        print(f"PURPOSE OF THIS RUN: {args.purpose}")
        print(f"Model: {models_to_run}")
        print(f"Dataset: {datasets_to_run}")
        print(f"Output: {output_detail_csv}, {output_timing_csv}, {output_summary_csv}")
        print("Starting model comparison (light -> heavy)...")
        df_results, df_timing = run_comparison(
            models_to_run, datasets_to_run, ATTACKS, run_purpose=args.purpose)

        # so multiple _run* files can be merged later and each row's run identified
        df_results.insert(0, "run_index", run_idx)
        df_timing.insert(0, "run_index", run_idx)

        df_results.to_csv(output_detail_csv, index=False)
        print(f"\nSaved per-call details: {output_detail_csv}")

        if not df_timing.empty:
            df_timing.to_csv(output_timing_csv, index=False)
            print(f"Saved per-dataset/model timing: {output_timing_csv}")

        df_summary = summarize_results(df_results, df_timing)
        if not df_summary.empty:
            df_summary.insert(0, "run_index", run_idx)
            df_summary.to_csv(output_summary_csv, index=False)
            print(f"Saved final summary table: {output_summary_csv}")

        print("\n" + "=" * 60)
        print("TIME PER DATASET (PER MODEL)")
        print("=" * 60)
        if not df_timing.empty:
            print(df_timing.to_string(index=False))

        print("\n" + "=" * 60)
        print("FINAL SUMMARY TABLE (all 3 datasets combined, sorted by speed)")
        print("=" * 60)
        if not df_summary.empty:
            print(df_summary.to_string(index=False))

    if args.runs > 1:
        print(f"\nFinished {args.runs} independent runs, results are in the "
              f"files suffixed _run1 .. _run{args.runs} for comparison.")
