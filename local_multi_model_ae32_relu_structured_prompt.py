"""
Variant of local_multi_model_ae32_relu.py: keeps the same AE backend (32-dim,
fixed nn.ReLU()) and the entire load/inference pipeline, only replaces the
PROMPT sent to the LLM - from a free-form 250-character format to a
STRUCTURED 5-field format (Source, Likely Cause, Affected Asset,
Recommendation, Risk Score) - so the operator can immediately read the
source/root cause/affected location/remediation instead of just a generic
sentence + score.

Direct comparison of the two prompt styles (real example, Smart Grid
dataset, "function code scan" attack, phi4-mini model):

  OLD PROMPT -> "The packet flow rate is significantly higher than normal,
                 indicating potential malicious activity. Risk Score: 8/10."
  NEW PROMPT -> "Source: 192.168.0.1
                 Likely Cause: Modbus flooding attack - unusually high packet
                   rate indicating potential DoS; consistent with a flood attack
                 Affected Asset: 192.168.0.31
                 Recommendation: Block source IP and monitor traffic; consider
                   increasing security measures to prevent further attacks
                 Risk Score: 9/10"

UPDATE (2026-09-08): runs 1-5 showed several models (e.g. qwen3:8b) writing a
vague "Likely Cause" ("elevated packet rate", "unusual function code") even
though the exact baseline/observed numbers were already in the prompt data -
they just weren't citing them. create_prompt() now explicitly requires the
Likely Cause sentence to cite the exact number(s) backing the claim (e.g.
"function code 40 ... 7.7x vs baseline"), with a GOOD/BAD example pair in the
prompt itself. This changes the prompt content, so any run from this point
on (run6+) is not directly comparable to runs 1-5's Likely Cause text quality
(risk_score/format_compliance/latency stayed on the same measurement, so
those columns are still comparable across all runs).

UPDATE (2026-09-08, same day, after run6): run6 showed a regression from the
above fix - the GOOD example originally used real-looking numbers ("function
code 40", "7.7x"), and 2 of the smaller/weaker models (phi4-mini 9/22 calls,
qwen3:8b 4/22 calls) just copy-pasted that example's numbers verbatim as
their answer, including on attacks whose actual data was completely
different (wrong function code, wrong ratio). Fixed by changing the example
to obviously-fake numbers ("function code 99", "12.3x") plus an explicit
"these are NOT this packet's data" warning, so the example can't be
mistaken for a valid literal answer. Not yet re-verified with a full run
(see run7+ once available).

Run standalone with:
    python local_multi_model_ae32_relu_structured_prompt.py --purpose "..." 2>&1 | tee local_multi_model_ae32_relu_structured_prompt/run_$(date +%Y%m%d_%H%M).log
(output CSVs also go into local_multi_model_ae32_relu_structured_prompt/,
created right at import time so a shell redirect into that folder works
from the very first run)

MODELS BEING TESTED (already pulled, running on a GTX 3060 GPU):
    phi4-mini, qwen3:4b, gemma4:e4b, qwen3:8b, openthinker:7b, deepseek-r1:8b,
    gemma4:12b, qwen3:14b

REQUIREMENTS BEFORE RUNNING:
1. Ollama installed and running (`ollama serve`).
2. The models to test already pulled (the script automatically skips any
   model that hasn't been pulled, see check_model_available()).
3. The 32-dim autoencoder's model/threshold files already present in the repo:
     <dataset>_ae32_relu_model.pt, <dataset>_threshold.txt
   (e.g. smart_grid_ae32_relu_model.pt, smart_grid_threshold.txt)

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
# the actual answer. The structured prompt also produces more completion
# tokens than the old prompt (~70-100 tokens instead of ~20-30, verified in
# practice with phi4-mini) but that's still far below 16384 so this value
# is kept as-is.
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
OUTPUT_DIR = Path("local_multi_model_ae32_relu_structured_prompt")
OUTPUT_DIR.mkdir(exist_ok=True)


local_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",       # Ollama doesn't check the key, but the SDK requires a value
    timeout=REQUEST_TIMEOUT_SEC,
)


# ============================================================
# 2. 32-DIM AUTOENCODER + PREPROCESSING PIPELINE (from the "Complete Pipeline" cell)
#    - IDENTICAL to local_multi_model_ae32_relu.py, nothing changed here.
# ============================================================

# CLASS:    AutoEncoder
# PURPOSE:  32-dim latent space autoencoder, fixed nn.ReLU() activation (not
#           tuned) - the ORIGINAL architecture from ics_simlab_sanh.ipynb
#           (cell "Autoencoder (AE)"). Must match 1:1 the architecture used
#           when training *_ae32_relu_model.pt (loaded via load_ae_model()) -
#           changing the architecture here without retraining would corrupt
#           the state_dict.
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


# ============================================================
# 2b. STRUCTURED PROMPT (differs from local_multi_model_ae32_relu.py)
# ============================================================

# Human-readable names for common Modbus function codes - included in the
# prompt instead of just the raw number, to help the model (especially a
# small one like phi4-mini) reason about the cause more accurately instead
# of having to recall the Modbus code table from memory.
MODBUS_FUNC_NAMES = {
    1: "Read Coils",
    2: "Read Discrete Inputs",
    3: "Read Holding Registers",
    4: "Read Input Registers",
    5: "Write Single Coil",
    6: "Write Single Register",
    15: "Write Multiple Coils",
    16: "Write Multiple Registers",
    22: "Mask Write Register",
    23: "Read/Write Multiple Registers",
    43: "Read Device Identification / Encapsulated Interface Transport",
}


def describe_func_code(code) -> str:
    code = int(code)
    name = MODBUS_FUNC_NAMES.get(code)
    if name:
        return f"{code} ({name})"
    return f"{code} (not a standard Modbus function code - may indicate function code scanning/fuzzing)"


def create_prompt(orig_packet_info, orig_flow_info):
    func_code_desc = (
        describe_func_code(orig_packet_info["modbus_function_code"])
        if pd.notna(orig_packet_info["modbus_function_code"]) else "N/A"
    )
    avg_rate = orig_flow_info["average_packet_rate"]
    rate_ratio = f"{orig_flow_info['packet_rate'] / avg_rate:.1f}x" if avg_rate else "N/A"

    return f'''
You are a SOC analyst reviewing a flagged anomalous packet on an Industrial
Control System (Modbus/TCP) network. Based on the data below, respond in
EXACTLY this format (one short line per field, no extra commentary):

Source: <suspected source IP/MAC, and whether it is internal or external to the local network>
Likely Cause: <one sentence citing the SPECIFIC evidence and its EXACT value(s) from the data below (e.g. the function code number, the rate ratio, the RTT) that support this conclusion - do not use vague words like "elevated", "unusual", or "high" without the number that backs it up>
Affected Asset: <the destination IP/device being affected>
Recommendation: <one specific action the operator should take right now>
Risk Score: X/10

GOOD Likely Cause style (fill in the placeholders below with THIS packet's own
values from the "Packet data" section further down - do not invent numbers,
and do not reuse any number shown elsewhere in this instructions section):
  "Non-standard function code <the function code number from Packet data>
   combined with a high packet rate (<the rate ratio from Packet data>x vs
   baseline) suggests active scanning or probing."
BAD Likely Cause style (too vague, do NOT write like this): "Unusual Modbus function code usage with elevated packet rate indicating potential reconnaissance."

--- Packet data ---
IP Source: {orig_packet_info["ip_src"]}
IP Destination: {orig_packet_info["ip_dst"]}
MAC Source: {orig_flow_info["most_common_mac_src"]}
MAC Destination: {orig_flow_info["most_common_mac_dst"]}
Protocol: {orig_packet_info["protocol"]}
Modbus Function Code: {func_code_desc}
Modbus PDU: {orig_packet_info["modbus_data"]}
Packet flow rate (last 4s window): {orig_flow_info["packet_rate"]} packets/sec
Baseline average packet rate (whole dataset): {avg_rate} packets/sec
Rate ratio vs baseline: {rate_ratio}
'''


def prepare_dataset(dataset_name):
    """Load the raw csv, run the already-trained 32-dim autoencoder, and
    return (df_orig, original_anomalies) for the risk-scoring step."""
    csv_path = find_dataset_csv(DATASET_FILENAMES[dataset_name])
    df_orig = pd.read_csv(csv_path)

    X_tensor, inference_indices, df_orig_with_indicies, df_orig_with_indicies_sampled = process_dl_dataset(df_orig)

    file_prefix = dataset_name.lower().replace(" ", "_")
    model_file = file_prefix + "_ae32_relu_model.pt"
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
    # See local_multi_model_ae32_relu.py for the detailed explanation: must
    # call the native /api/chat endpoint (not OpenAI-compat) for
    # options/think to actually be applied on this Ollama build.
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


def extract_field(text: str, field_name: str):
    """Extract the content after a 'Field name: value' field on a single
    line, shared by all 5 fields of the structured prompt (Source, Likely
    Cause, Affected Asset, Recommendation, Risk Score)."""
    if not text:
        return None
    text = strip_think_block(text)
    match = re.search(rf"{re.escape(field_name)}:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_source(text: str):
    return extract_field(text, "Source")


def extract_likely_cause(text: str):
    return extract_field(text, "Likely Cause")


def extract_affected_asset(text: str):
    return extract_field(text, "Affected Asset")


def extract_recommendation(text: str):
    return extract_field(text, "Recommendation")


def extract_risk_score(text: str):
    value = extract_field(text, "Risk Score")
    if not value:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", value)
    return float(match.group(1)) if match else None


def check_format_compliance(text: str):
    """Unlike local_multi_model_ae32_relu.py (which only checks for a Risk
    Score + a character-length limit): the new prompt has no character
    limit, instead checking that all 5 required fields are present."""
    text = strip_think_block(text)
    fields = {
        "has_source": extract_source(text) is not None,
        "has_likely_cause": extract_likely_cause(text) is not None,
        "has_affected_asset": extract_affected_asset(text) is not None,
        "has_recommendation": extract_recommendation(text) is not None,
        "has_risk_score": extract_risk_score(text) is not None,
    }
    fields["all_fields_present"] = all(fields.values())
    return fields


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

                    if output:
                        format_fields = check_format_compliance(output)
                    else:
                        format_fields = {
                            "has_source": False, "has_likely_cause": False,
                            "has_affected_asset": False, "has_recommendation": False,
                            "has_risk_score": False, "all_fields_present": False,
                        }

                    results.append({
                        "run_purpose": run_purpose,
                        "dataset": dataset_name,
                        "model": model_name,
                        "attack_type": attack_name,
                        "attack_specific": attack_specific,
                        "repeat": repeat_idx + 1,
                        "prompt": prompt,
                        "output": output,
                        "source": extract_source(output) if output else None,
                        "likely_cause": extract_likely_cause(output) if output else None,
                        "affected_asset": extract_affected_asset(output) if output else None,
                        "recommendation": extract_recommendation(output) if output else None,
                        "risk_score": extract_risk_score(output) if output else None,
                        "format_all_fields_present": format_fields["all_fields_present"],
                        "latency_sec": round(latency, 2),
                        "prompt_tokens": usage.prompt_tokens if usage else None,
                        "completion_tokens": usage.completion_tokens if usage else None,
                        "error": error,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    })

                    if output:
                        risk_score = extract_risk_score(output)
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
            valid["format_all_fields_present"].mean() * 100
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
        description="Compare risk-scoring (structured prompt) across several local LLMs on the ICS-SimLab dataset.")
    parser.add_argument(
        "--purpose", required=True,
        help="Required: a short description of this run's purpose, printed to "
             "the log and stored in the run_purpose CSV column, for later "
             "aggregation of experiments.")
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
        help="Suffix added to the output CSV filenames (e.g. 'test1') so it "
             "doesn't overwrite a previous full run's results. Leave blank "
             "to use the default filenames.")
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of times to re-run the ENTIRE pipeline (each run calls "
             "the LLM again from scratch) to compare consistency across "
             "independent runs. Each run writes its own CSV files "
             "(suffixed _run1, _run2, ...). Default 1.")
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
