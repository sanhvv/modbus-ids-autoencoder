"""
So sánh các local LLM (chạy qua Ollama) cho pipeline risk-scoring ICS-IDS.

YÊU CẦU TRƯỚC KHI CHẠY:
1. Đã cài Ollama và chạy `ollama serve` trên máy local (không phải Colab).
2. Đã pull các model cần test, ví dụ:
     ollama pull phi4-mini
     ollama pull qwen3:4b
     ollama pull qwen3:8b
     ollama pull deepseek-r1:8b
3. Chạy đoạn code này TRONG CÙNG notebook/kernel đã load sẵn:
   - autoencoder (`autoencoder`, hoặc lặp qua `ae_models_test`)
   - `original_anomalies`, `df_orig`
   - các hàm: select_anomalous_packet, extract_packet_info, create_prompt,
              attacks (dict)
   Nếu chưa có, script sẽ báo lỗi rõ ràng ở bước kiểm tra đầu tiên.
"""

import re
import time
import json
import statistics
from datetime import datetime

import pandas as pd
from openai import OpenAI

# ============================================================
# 1. CẤU HÌNH
# ============================================================

# Danh sách model theo thứ tự NHẸ -> NẶNG (để phát hiện sớm nếu máy không
# kham nổi model nặng, dừng lại ở mức phù hợp thay vì chờ lâu vô ích)
MODELS_TO_TEST = [
    "phi4-mini",        # ~3.8B, nhẹ nhất, chạy tốt thuần CPU
    "qwen3:4b",          # ~4B, đa ngôn ngữ tốt
    "qwen3:8b",          # ~8B, cân bằng chất lượng/tốc độ
    "deepseek-r1:8b",   # ~8B, thiên về reasoning từng bước
    # "gemma4:latest",   # bật nếu máy có >=24GB RAM/unified memory
    # "gpt-oss:20b",     # bật nếu máy có >=24GB RAM/unified memory
]

# Số lần lặp lại MỖI prompt để đo độ ổn định (consistency) của risk score
REPEATS_PER_PROMPT = 3

# Timeout mỗi lần gọi model (giây) - model nặng trên CPU có thể rất chậm
REQUEST_TIMEOUT_SEC = 300

OLLAMA_BASE_URL = "http://localhost:11434/v1"

OUTPUT_CSV = "local_llm_comparison_results.csv"
OUTPUT_SUMMARY_CSV = "local_llm_comparison_summary.csv"


# ============================================================
# 2. CLIENT VÀ HÀM GỌI MODEL
# ============================================================

local_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",       # Ollama không kiểm tra key, nhưng SDK bắt buộc phải có giá trị
    timeout=REQUEST_TIMEOUT_SEC,
)


def check_model_available(client, model_name: str) -> bool:
    """Kiểm tra model đã được `ollama pull` chưa, tránh chờ lâu rồi mới báo lỗi."""
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        return any(model_name in m for m in available)
    except Exception as e:
        print(f"  [CẢNH BÁO] Không kết nối được Ollama server: {e}")
        return False


def use_llm_local(client, prompt: str, model_name: str):
    """Gọi model local qua Chat Completions API (Ollama không hỗ trợ Responses API)."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,   # thấp để risk score ổn định giữa các lần lặp
    )
    output = response.choices[0].message.content
    usage = response.usage
    return output, usage


def strip_think_block(text: str) -> str:
    """Bỏ block <think>...</think> của reasoning model (vd deepseek-r1) trước khi
    đo format compliance, để so sánh công bằng với các model khác."""
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_risk_score(text: str):
    """Trích số risk score từ output, trả về None nếu không tìm thấy đúng format."""
    if not text:
        return None
    text = strip_think_block(text)
    match = re.search(r"Risk Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def check_format_compliance(text: str, max_chars: int = 250):
    """Kiểm tra output có tuân thủ đúng yêu cầu format trong prompt gốc không."""
    text = strip_think_block(text)
    has_risk_score = extract_risk_score(text) is not None
    # Đo độ dài phần văn bản trước "Risk Score:" (đúng ý yêu cầu prompt gốc)
    body = re.split(r"Risk Score:", text, flags=re.IGNORECASE)[0] if text else ""
    within_length = len(body.strip()) <= max_chars
    return has_risk_score, within_length


# ============================================================
# 3. VÒNG LẶP CHÍNH: chạy từng model, từng attack type, N lần lặp
# ============================================================

def run_comparison(models_to_test, attacks, original_anomalies, df_orig):
    results = []

    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        if not check_model_available(local_client, model_name):
            print(f"  [BỎ QUA] Model '{model_name}' chưa sẵn sàng trên Ollama. "
                  f"Chạy: ollama pull {model_name}")
            continue

        for attack_specific, attack_name in attacks.items():
            # chọn packet đại diện (tái dùng hàm đã có trong notebook gốc)
            original_packet = select_anomalous_packet(
                original_anomalies, df_orig, attack_specific)
            orig_packet_info, orig_flow_info = extract_packet_info(
                original_packet, df_orig)
            prompt = create_prompt(orig_packet_info, orig_flow_info)

            risk_scores_this_prompt = []

            for repeat_idx in range(REPEATS_PER_PROMPT):
                print(f"  [{attack_name}] lần {repeat_idx + 1}/{REPEATS_PER_PROMPT}...",
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
                    print(f"LỖI: {error}")

                risk_score = extract_risk_score(output) if output else None
                has_risk_score, within_length = (
                    check_format_compliance(output) if output else (False, False)
                )
                if risk_score is not None:
                    risk_scores_this_prompt.append(risk_score)

                results.append({
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

            # in nhanh độ lệch risk score cho attack này
            if len(risk_scores_this_prompt) > 1:
                spread = max(risk_scores_this_prompt) - min(risk_scores_this_prompt)
                print(f"    -> Độ lệch risk score qua {REPEATS_PER_PROMPT} lần: {spread:.1f}")

    return pd.DataFrame(results)


# ============================================================
# 4. TỔNG HỢP KẾT QUẢ THÀNH BẢNG SO SÁNH
# ============================================================

def summarize_results(df_results: pd.DataFrame) -> pd.DataFrame:
    if df_results.empty:
        print("Không có kết quả nào (có thể không model nào sẵn sàng trên Ollama).")
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

        # độ ổn định: std của risk score, gộp theo từng attack rồi lấy trung bình
        consistency_scores = []
        for attack_type, attack_group in valid.groupby("attack_type"):
            scores = attack_group["risk_score"].dropna().tolist()
            if len(scores) > 1:
                consistency_scores.append(statistics.pstdev(scores))
        avg_std_risk_score = (
            statistics.mean(consistency_scores) if consistency_scores else None
        )

        avg_completion_tokens = (
            valid["completion_tokens"].mean() if n_success > 0 else None
        )

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
        })

    return pd.DataFrame(summary_rows).sort_values("avg_latency_sec")


# ============================================================
# 5. CHẠY
# ============================================================

if __name__ == "__main__":
    # Các biến sau PHẢI đã tồn tại trong notebook gốc trước khi chạy script này:
    #   attacks, original_anomalies, df_orig,
    #   select_anomalous_packet, extract_packet_info, create_prompt
    required_vars = [
        "attacks", "original_anomalies", "df_orig",
        "select_anomalous_packet",
        "extract_packet_info", "create_prompt",
    ]
    missing = [v for v in required_vars if v not in dir()]
    if missing:
        raise RuntimeError(
            f"Thiếu biến/hàm cần thiết từ notebook gốc: {missing}. "
            f"Hãy chạy các cell load dataset + định nghĩa hàm trước khi chạy script này."
        )

    print("Bắt đầu so sánh model (từ nhẹ -> nặng)...")
    df_results = run_comparison(MODELS_TO_TEST, attacks, original_anomalies, df_orig)

    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nĐã lưu chi tiết từng lần chạy: {OUTPUT_CSV}")

    df_summary = summarize_results(df_results)
    if not df_summary.empty:
        df_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
        print(f"Đã lưu bảng tổng hợp: {OUTPUT_SUMMARY_CSV}")

    print("\n" + "=" * 60)
    print("BẢNG SO SÁNH (sắp xếp theo tốc độ, nhanh nhất trước)")
    print("=" * 60)
    print(df_summary.to_string(index=False))