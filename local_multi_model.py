"""
So sanh risk-scoring giua nhieu local LLM (chay qua Ollama) tren CA 3 dataset
ICS-SimLab (Intelligent Electronic Device, Smart Grid, Water Bottle Factory).

Day la ban standalone, mo rong tu "local llms.py" (chi chay 1 dataset, phai
chay trong notebook da co san bien/ham) - script nay tu load + clean dataset,
tu load autoencoder 32-dim va nguong (threshold) da luu san cho ca 3 dataset,
roi moi goi cac model local de risk-scoring, nen co the chay doc lap bang:
    python local_multi_model.py

CAC MODEL DUOC TEST (dang duoc pull ve, chay tren GPU GTX 3060):
    phi4-mini, qwen3:4b, gemma4:e4b, qwen3:8b, openthinker:7b, deepseek-r1:8b,
    gemma4:12b, qwen3:14b

YEU CAU TRUOC KHI CHAY:
1. Da cai va dang chay Ollama (`ollama serve`).
2. Da pull cac model can test (script se TU DONG bo qua model nao chua pull,
   xem check_model_available()).
3. Cac file model/threshold cua autoencoder 32-dim da co san trong repo:
     <dataset>_ae_model.pt, <dataset>_threshold.txt
   (vd: smart_grid_ae_model.pt, smart_grid_threshold.txt)

CANH BAO TOC DO: so lan goi model = so_model x so_dataset x so_attack_type x
REPEATS_PER_PROMPT. Mac dinh REPEATS_PER_PROMPT=1 de chay thu nhanh; tang len
2-3 neu muon do do on dinh (consistency) cua risk score.
"""

import re
import os
import sys
import time
import argparse
import statistics
from datetime import datetime

import pandas as pd
import requests
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from openai import OpenAI

from retrain_ae_9dim import DATA_DIR, DATASET_FILENAMES, find_dataset_csv

# ============================================================
# 1. CAU HINH
# ============================================================

# Danh sach model theo thu tu NHE -> NANG. Chay tren GPU GTX 3060.
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

# Attack type -> ten hien thi (giong "Complete Pipeline" cell trong notebook)
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

REPEATS_PER_PROMPT = 1      # tang len de do do on dinh cua risk score
REQUEST_TIMEOUT_SEC = 600   # model 12B/14B tren GPU 3060 co the van can vai chuc giay
# Ollama mac dinh chay request voi context window 4096 token neu khong set
# num_ctx. Cac model "thinking" (gemma4:12b, qwen3, deepseek-r1, ...) sinh
# khoi <think>...</think> truoc khi tra loi va co the tieu het toan bo 4096
# token do cho phan think, bi cat ngang truoc khi kip sinh cau tra loi that
# -> response.choices[0].message.content rong, khong loi, khong risk score
# (xem local_multi_model_results_1.csv, cot gemma4:12b: completion_tokens
# ~3800 nhung output/risk_score rong o ca 3 dataset). Tang num_ctx de model
# co du cho vua think vua tra loi.
NUM_CTX = 16384
# May nay chay 2 Ollama instance: mac dinh (11434, model dung chung/khong lien
# quan) va instance rieng cua user (11435, noi cac model o MODELS_TO_TEST duoc
# pull vao) - phai tro dung port 11435, khong dung mac dinh.
OLLAMA_BASE_URL = "http://localhost:11435/v1"
# Dung cho check_model_available() (list model qua OpenAI-compat, khong bi
# anh huong boi bug num_ctx o tren). use_llm_local() goi thang endpoint goc
# ben duoi (khong co "/v1") de options.num_ctx duoc ap dung dung.
OLLAMA_NATIVE_BASE_URL = OLLAMA_BASE_URL.removesuffix("/v1")

OUTPUT_DETAIL_CSV = "local_multi_model_results.csv"
OUTPUT_DATASET_TIMING_CSV = "local_multi_model_dataset_timing.csv"
OUTPUT_SUMMARY_CSV = "local_multi_model_summary.csv"


local_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",       # Ollama khong kiem tra key, nhung SDK bat buoc phai co gia tri
    timeout=REQUEST_TIMEOUT_SEC,
)


# ============================================================
# 2. AUTOENCODER 32-DIM + PIPELINE TIEN XU LY (tu cell "Complete Pipeline")
# ============================================================

# CLASS:    AutoEncoder
# PURPOSE:  Autoencoder 32-dim latent space dung de phat hien goi tin bat thuong.
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
    """Load raw csv, chay autoencoder 32-dim da train san, tra ve
    (df_orig, original_anomalies) de dung cho buoc risk-scoring."""
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
# 3. GOI MODEL LOCAL (tu "local llms.py")
# ============================================================

def check_model_available(client, model_name: str) -> bool:
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        return any(model_name in m for m in available)
    except Exception as e:
        print(f"  [CANH BAO] Khong ket noi duoc Ollama server: {e}")
        return False


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def use_llm_local(client, prompt: str, model_name: str):
    # QUAN TRONG: goi thang endpoint goc /api/chat cua Ollama bang requests,
    # KHONG dung client OpenAI-compat (client.chat.completions.create) o day.
    # Da kiem chung bang tay (curl + `ollama ps`) rang endpoint OpenAI-compat
    # /v1/chat/completions cua ban Ollama dang chay (0.32.14) AM THAM BO QUA
    # "options": {"num_ctx": ...} (ca dang long trong "options" lan dang phang
    # "num_ctx" o top-level) - model luon duoc load lai voi num_ctx=4096 mac
    # dinh du client gui gi di nua (ollama ps van bao context_length=4096).
    # Endpoint goc /api/chat thi ap dung dung (ollama ps bao dung context_length
    # + size_vram tang tuong ung). Day chinh la ly do lan fix dau tien (dung
    # extra_body qua client OpenAI-compat) khong co tac dung: gemma4:12b van bi
    # cat ngang o dung 4096 token (prompt+completion) nhu truoc khi fix.
    #
    # "think": False - README truoc ghi "da thu think:false, khong cach nao
    # tat duoc" nhung do la thu qua duong OpenAI-compat (cung bi bug nhu tren,
    # option bi lo am tham). Qua endpoint goc, think:false hoat dong dung: model
    # tra loi thang khong sinh khoi <think>/field "thinking", nhanh hon nhieu
    # (vd gemma4:12b: ~2-5s thay vi hang tram giay hoac rong hoan toan).
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
        # response.raise_for_status() chi bao status line (vd "400 Client Error:
        # Bad Request for url: ..."), KHONG kem body - ma body moi la noi Ollama
        # ghi ly do that su (vd het VRAM, model runner crash...). Dinh kem body
        # vao message de lan sau debug duoc ngay, khong phai doan mo.
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
# 4. VONG LAP CHINH: tung dataset -> tung model -> tung attack type -> N lan lap
# ============================================================

def build_prompts_for_dataset(df_orig, original_anomalies, attacks):
    """Tao san 1 prompt cho moi attack type (dung chung cho tat ca model/repeat
    de dam bao so sanh cong bang tren cung 1 goi tin duoc chon)."""
    prompts = {}
    for attack_specific, attack_name in attacks.items():
        if not (original_anomalies["attack_specific"] == attack_specific).any():
            print(f"  [BO QUA] Khong co anomaly nao duoc AE phat hien cho attack '{attack_name}' o dataset nay.")
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
                print(f"  [BO QUA] Model '{model_name}' chua san sang tren Ollama. "
                      f"Chay: ollama pull {model_name}")
                continue

            dataset_model_start = time.time()

            for attack_specific, (attack_name, prompt) in prompts.items():
                for repeat_idx in range(REPEATS_PER_PROMPT):
                    print(f"  [{attack_name}] lan {repeat_idx + 1}/{REPEATS_PER_PROMPT}...",
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
                        print(f"LOI: {error}")

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
            print(f"  -> Thoi gian chay model '{model_name}' tren dataset '{dataset_name}': "
                  f"{dataset_model_time:.2f}s ({n_calls} lan goi)")

    return pd.DataFrame(results), pd.DataFrame(dataset_timing)


# ============================================================
# 5. TONG HOP KET QUA CUOI CUNG (gop ca 3 dataset, theo tung model)
# ============================================================

def summarize_results(df_results: pd.DataFrame, df_timing: pd.DataFrame) -> pd.DataFrame:
    if df_results.empty:
        print("Khong co ket qua nao (co the khong model nao san sang tren Ollama).")
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
# 6. CHAY
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="So sanh risk-scoring giua nhieu local LLM tren dataset ICS-SimLab.")
    parser.add_argument(
        "--purpose", required=True,
        help="Bat buoc: mo ta ngan gon muc dich lan chay nay (vd 'rerun gemma4:12b "
             "sau khi fix num_ctx'), duoc ghi vao log va cot run_purpose trong CSV "
             "de sau nay tong hop lai cac experiment.")
    parser.add_argument(
        "--models", default=None,
        help=f"Danh sach model can chay, cach nhau bang dau phay (mac dinh: ca "
             f"{len(MODELS_TO_TEST)} model). Vd: --models gemma4:12b")
    parser.add_argument(
        "--datasets", default=None,
        help=f"Danh sach dataset can chay, cach nhau bang dau phay (mac dinh: ca 3 "
             f"dataset). Vd: --datasets 'Smart Grid,Water Bottle Factory'")
    parser.add_argument(
        "--tag", default=None,
        help="Hau to them vao ten file CSV output (vd 'gemma4_ctxfix') de khong "
             "de len ket qua lan chay day du truoc do. Bo trong = dung ten file mac dinh.")
    parser.add_argument(
        "--runs", type=int, default=1,
        help="So lan chay lai TOAN BO pipeline (moi lan goi lai LLM tu dau, khong "
             "phai lap lai cung 1 cau tra loi) de so sanh do on dinh giua cac lan "
             "chay doc lap. Moi lan ghi ra file CSV rieng (hau to _run1, _run2, ...). "
             "Mac dinh 1 (chi chay 1 lan, ten file giu nguyen nhu truoc, khong them hau to).")
    args = parser.parse_args()
    if args.runs < 1:
        sys.exit("Loi: --runs phai >= 1")
    return args


if __name__ == "__main__":
    args = parse_args()

    models_to_run = MODELS_TO_TEST
    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in requested if m not in MODELS_TO_TEST]
        if unknown:
            sys.exit(f"Loi: model khong ton tai trong MODELS_TO_TEST: {unknown}")
        models_to_run = requested

    datasets_to_run = DATASETS
    if args.datasets:
        requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
        unknown = [d for d in requested if d not in DATASETS]
        if unknown:
            sys.exit(f"Loi: dataset khong ton tai trong DATASETS: {unknown}")
        datasets_to_run = requested

    base_suffix = f"_{args.tag}" if args.tag else ""

    for run_idx in range(1, args.runs + 1):
        run_suffix = base_suffix + (f"_run{run_idx}" if args.runs > 1 else "")
        output_detail_csv = OUTPUT_DETAIL_CSV.replace(".csv", f"{run_suffix}.csv")
        output_timing_csv = OUTPUT_DATASET_TIMING_CSV.replace(".csv", f"{run_suffix}.csv")
        output_summary_csv = OUTPUT_SUMMARY_CSV.replace(".csv", f"{run_suffix}.csv")

        if args.runs > 1:
            print(f"\n{'#'*60}\nLAN CHAY {run_idx}/{args.runs}\n{'#'*60}")
        print(f"MUC DICH LAN CHAY NAY: {args.purpose}")
        print(f"Model: {models_to_run}")
        print(f"Dataset: {datasets_to_run}")
        print(f"Output: {output_detail_csv}, {output_timing_csv}, {output_summary_csv}")
        print("Bat dau so sanh model (tu nhe -> nang)...")
        df_results, df_timing = run_comparison(
            models_to_run, datasets_to_run, ATTACKS, run_purpose=args.purpose)

        # de sau nay gop nhieu file _run* lai va biet dong nao thuoc lan chay nao
        df_results.insert(0, "run_index", run_idx)
        df_timing.insert(0, "run_index", run_idx)

        df_results.to_csv(output_detail_csv, index=False)
        print(f"\nDa luu chi tiet tung lan chay: {output_detail_csv}")

        if not df_timing.empty:
            df_timing.to_csv(output_timing_csv, index=False)
            print(f"Da luu thoi gian chay theo tung dataset/model: {output_timing_csv}")

        df_summary = summarize_results(df_results, df_timing)
        if not df_summary.empty:
            df_summary.insert(0, "run_index", run_idx)
            df_summary.to_csv(output_summary_csv, index=False)
            print(f"Da luu bang tong hop cuoi cung: {output_summary_csv}")

        print("\n" + "=" * 60)
        print("THOI GIAN CHAY THEO TUNG DATASET (tung model)")
        print("=" * 60)
        if not df_timing.empty:
            print(df_timing.to_string(index=False))

        print("\n" + "=" * 60)
        print("BANG TONG HOP CUOI CUNG (gop ca 3 dataset, sap xep theo toc do)")
        print("=" * 60)
        if not df_summary.empty:
            print(df_summary.to_string(index=False))

    if args.runs > 1:
        print(f"\nDa chay xong {args.runs} lan doc lap, ket qua nam trong cac file "
              f"co hau to _run1 .. _run{args.runs} de so sanh.")
