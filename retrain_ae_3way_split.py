"""
Retrains the 32-dim autoencoder (same architecture as local_multi_model_ae32_relu.py:
nn.Linear + fixed nn.ReLU, encoder 128->64->32, decoder 32->64->128) using a
proper 3-way split (train/validation/test) instead of the leaky methodology
found in the original notebook and in the other tuning scripts in this repo.

WHY THIS SCRIPT EXISTS
-----------------------
The original AE training (ics_simlab_sanh.ipynb, cell "Autoencoder (AE)") and
every downstream evaluation function in this repo (detect_anomaly_9dim,
evaluate_config, eval_ae_9dim, ...) computes the anomaly threshold from the
SAME normal data the model was trained on, then evaluates on a set that
still includes those same training rows. This is a form of data leakage:
reconstruction error on training data is typically lower (more optimistic)
than on genuinely unseen normal data, so both the threshold and the
reported precision/recall/F1 can look better than what the model would
actually achieve on new, never-seen traffic.

This script fixes that by splitting the NORMAL data three ways:
  - train      (default 70%) -> only this is used to fit the autoencoder
  - validation (default 15%) -> only this is used to compute the anomaly
                                 threshold (95th percentile of reconstruction
                                 error), never touched during training
  - test       (default 15%) -> combined with ALL attack rows (never used
                                 for training or thresholding) to produce
                                 an honest, held-out evaluation

For direct comparison, it ALSO reproduces the original "leaky" methodology
on the exact same trained model (threshold from train data, evaluate on
everything) and reports both side by side, so the size of the leakage-driven
optimism gap is visible per dataset.

Output (model .pt, threshold .txt, results CSV) goes into
retrain_ae_3way_split/. Run in a screen session, e.g.:
    screen -S ae_3way
    python retrain_ae_3way_split.py 2>&1 | tee retrain_ae_3way_split/run_$(date +%Y%m%d_%H%M).log
    # Ctrl-A D to detach
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from retrain_ae_9dim import load_and_clean_datasets

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15   # must sum to 1.0 with the two above

EVAL_PERCENTILE = 95
RANDOM_SEED = 42

EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# This script's output (model .pt, threshold .txt, results CSV, log) all go
# here - created right at import time so a shell log redirect into this
# folder works from the very first run.
OUTPUT_DIR = Path("retrain_ae_3way_split")
OUTPUT_DIR.mkdir(exist_ok=True)


# CLASS:    AutoEncoder
# PURPOSE:  Identical architecture to the one in local_multi_model_ae32_relu.py
#           (32-dim latent, fixed nn.ReLU()) - kept as a separate definition
#           here (not imported) since this script trains its own fresh
#           weights rather than loading the existing *_ae_model.pt files.
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

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
        return self.decoder(self.encoder(x))


def preprocess(df):
    """Convert attack_specific into a binary is_attack label and split into
    normal-only and attack-only feature frames (no train/test split yet -
    that happens next, only on the normal rows)."""
    df = df.copy()
    if "is_attack" not in df.columns:
        df["is_attack"] = (df["attack_specific"] != 0).astype(int)
        if "attack_specific" in df.columns:
            df = df.drop(columns=["attack_specific"])

    df_normal = df[df["is_attack"] == 0].drop(columns=["is_attack"])
    df_attack = df[df["is_attack"] == 1].drop(columns=["is_attack"])
    return df_normal, df_attack


def three_way_split_normal(df_normal, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC,
                            test_frac=TEST_FRAC, seed=RANDOM_SEED):
    """Split ONLY the normal rows into train/val/test. Attack rows are never
    part of this split - they go entirely into the test set later, since
    they must never influence training or threshold selection."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, \
        "TRAIN_FRAC + VAL_FRAC + TEST_FRAC must sum to 1.0"

    df_train, df_rest = train_test_split(
        df_normal, train_size=train_frac, random_state=seed, shuffle=True)
    relative_val_frac = val_frac / (val_frac + test_frac)
    df_val, df_test = train_test_split(
        df_rest, train_size=relative_val_frac, random_state=seed, shuffle=True)

    return df_train, df_val, df_test


def train_autoencoder(X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    input_dim = X_train_t.shape[1]
    model = AutoEncoder(input_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0

        for i in range(0, X_train_t.size(0), batch_size):
            batch = X_train_t[i:i + batch_size]
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X_train_t):.6f}, Time: {epoch_time:.2f}s")

    return model


def reconstruction_error(model, X: pd.DataFrame) -> np.ndarray:
    X_t = torch.tensor(X.values, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_t)
        mse = torch.mean(torch.pow(X_t - reconstructed, 2), dim=1).numpy()
    return mse


def evaluate(model, X_eval: pd.DataFrame, y_true: np.ndarray, threshold: float) -> dict:
    errors = reconstruction_error(model, X_eval)
    y_pred = (errors > threshold).astype(int)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "threshold": threshold,
        "n_normal": int((y_true == 0).sum()),
        "n_attack": int((y_true == 1).sum()),
        "precision_attack": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_attack": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_attack": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_weighted": report["weighted avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "confusion_matrix": cm.tolist(),
    }


def run_for_dataset(dataset_name: str, df_clean: pd.DataFrame) -> dict:
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    dataset_start = time.time()

    df_normal, df_attack = preprocess(df_clean)
    df_train, df_val, df_test_normal = three_way_split_normal(df_normal)

    print(f"Normal rows: {len(df_normal)} total -> "
          f"train={len(df_train)}, val={len(df_val)}, test={len(df_test_normal)}")
    print(f"Attack rows: {len(df_attack)} (never used for training or thresholding)")

    model = train_autoencoder(df_train)

    file_prefix = dataset_name.lower().replace(" ", "_")
    torch.save(model.state_dict(), OUTPUT_DIR / f"{file_prefix}_ae_model_3way.pt")

    # --- Proper (honest) evaluation ---
    # Threshold from VALIDATION normal data (never seen during training).
    val_errors = reconstruction_error(model, df_val)
    proper_threshold = float(np.percentile(val_errors, EVAL_PERCENTILE))

    with open(OUTPUT_DIR / f"{file_prefix}_threshold_3way.txt", "w") as f:
        f.write(str(proper_threshold))

    # Test set = held-out normal test rows + ALL attack rows (both unseen).
    X_test = pd.concat([df_test_normal, df_attack])
    y_test = np.array([0] * len(df_test_normal) + [1] * len(df_attack))
    proper_result = evaluate(model, X_test, y_test, proper_threshold)

    print(f"\n[PROPER 3-way split] threshold={proper_threshold:.6f}")
    print(f"  precision_attack={proper_result['precision_attack']:.4f}, "
          f"recall_attack={proper_result['recall_attack']:.4f}, "
          f"f1_attack={proper_result['f1_attack']:.4f}")
    print(f"  confusion matrix: {proper_result['confusion_matrix']}")

    # --- Leaky evaluation (reproduces the ORIGINAL notebook's methodology,
    # same trained model, only the threshold/eval-set logic differs) ---
    # Threshold from TRAINING data itself.
    train_errors = reconstruction_error(model, df_train)
    leaky_threshold = float(np.percentile(train_errors, EVAL_PERCENTILE))

    # "Test" set = the ENTIRE normal population (including the exact rows
    # used for training) + all attack rows - matches preprocess_ae()'s
    # X_test = X in the original notebook.
    X_leaky = pd.concat([df_normal, df_attack])
    y_leaky = np.array([0] * len(df_normal) + [1] * len(df_attack))
    leaky_result = evaluate(model, X_leaky, y_leaky, leaky_threshold)

    print(f"\n[LEAKY - original methodology] threshold={leaky_threshold:.6f}")
    print(f"  precision_attack={leaky_result['precision_attack']:.4f}, "
          f"recall_attack={leaky_result['recall_attack']:.4f}, "
          f"f1_attack={leaky_result['f1_attack']:.4f}")
    print(f"  confusion matrix: {leaky_result['confusion_matrix']}")

    dataset_time = time.time() - dataset_start
    print(f"\nDataset {dataset_name} total time: {dataset_time:.2f}s")

    return {
        "dataset": dataset_name,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_test_normal": len(df_test_normal),
        "n_attack": len(df_attack),
        "proper": proper_result,
        "leaky": leaky_result,
        "dataset_time_s": dataset_time,
    }


def build_comparison_table(all_results: list) -> pd.DataFrame:
    rows = []
    for r in all_results:
        for method in ("proper", "leaky"):
            res = r[method]
            rows.append({
                "dataset": r["dataset"],
                "method": "3-way split (honest)" if method == "proper" else "leaky (original)",
                "threshold": round(res["threshold"], 6),
                "precision_attack": round(res["precision_attack"], 4),
                "recall_attack": round(res["recall_attack"], 4),
                "f1_attack": round(res["f1_attack"], 4),
                "f1_weighted": round(res["f1_weighted"], 4),
                "accuracy": round(res["accuracy"], 4),
                "n_eval_normal": res["n_normal"],
                "n_eval_attack": res["n_attack"],
            })
    return pd.DataFrame(rows)


def main():
    ics_datasets = load_and_clean_datasets()

    all_results = []
    total_time = 0.0

    for dataset_name, df_clean in ics_datasets.items():
        result = run_for_dataset(dataset_name, df_clean)
        all_results.append(result)
        total_time += result["dataset_time_s"]

    comparison = build_comparison_table(all_results)
    comparison.to_csv(OUTPUT_DIR / "leaky_vs_3way_comparison.csv", index=False)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON: leaky (original) vs 3-way split (honest)")
    print("=" * 60)
    print(comparison.to_string(index=False))

    print("\nOptimism gap (leaky f1_attack - honest f1_attack) per dataset:")
    for dataset_name in ics_datasets:
        sub = comparison[comparison["dataset"] == dataset_name]
        leaky_f1 = sub[sub["method"] == "leaky (original)"]["f1_attack"].iloc[0]
        proper_f1 = sub[sub["method"] == "3-way split (honest)"]["f1_attack"].iloc[0]
        print(f"  {dataset_name}: leaky={leaky_f1:.4f}, honest={proper_f1:.4f}, "
              f"gap={leaky_f1 - proper_f1:+.4f}")

    print(f"\nSaved comparison table: {OUTPUT_DIR / 'leaky_vs_3way_comparison.csv'}")
    print(f"Total time consumed for all datasets: {total_time:.2f}s")


if __name__ == "__main__":
    main()
