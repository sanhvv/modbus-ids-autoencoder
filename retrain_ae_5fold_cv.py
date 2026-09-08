"""
5-fold cross-validation stability analysis across all 3 autoencoder
architectures used in this repo: 32-dim Linear+ReLU (retrain_ae_3way_split.py
/ local_multi_model_ae32_relu.py), 9-dim Linear+ReLU (retrain_ae_9dim.py), and
the VAE with per-dataset best tuned hyperparameters (retrain_ae_vae_best.py).

For each dataset, the full cleaned dataframe is split into 5 disjoint
subsets ("subset1".."subset5") via KFold. For each of the 5 folds:
  - the 4 non-held-out subsets are combined, and only their NORMAL rows are
    used to train (the anomaly-detection convention used everywhere in this
    repo: the model never sees attack rows during training);
  - the ENTIRE held-out subset (both normal and attack rows, in whatever
    proportion naturally falls into it - not downsampled/balanced) is used
    as the test set.
This is repeated for all 5 folds x all 3 architectures, and the per-fold
metrics are compared (mean/std across folds) to see how much each
architecture's performance depends on which subset was held out, i.e. how
stable it is - not just its single-split performance.

Runtime note: the 32-dim and 9-dim Linear architectures both train for 30
epochs per fold (same as retrain_ae_3way_split.py / retrain_ae_9dim.py), and
this now runs 5x per dataset (once per fold) instead of once - on the
Intelligent Electronic Device dataset alone, a single 30-epoch run over
~156k normal rows took ~1000s in retrain_ae_3way_split.py, so this script's
full run across 3 datasets x 5 folds x 3 architectures is expected to take
several hours. The VAE architecture is much faster per epoch and does not
dominate the runtime. Run in a screen session, e.g.:
    screen -S ae_5fold_cv
    python retrain_ae_5fold_cv.py 2>&1 | tee retrain_ae_5fold_cv/run_$(date +%Y%m%d_%H%M).log
    # Ctrl-A D to detach

Does NOT save per-fold model .pt files (5 folds x 3 architectures x 3
datasets = 45 models) - the goal here is comparing performance across folds
for stability, not producing a deployable model. See retrain_ae_vae_best.py /
retrain_ae_9dim.py / the "Autoencoder (AE)" cell in ics_simlab_sanh.ipynb for
scripts that save a final trained model for actual use.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

from retrain_ae_9dim import load_and_clean_datasets, construct_ae_9dim
from retrain_ae_3way_split import train_autoencoder as train_ae32
from retrain_ae_vae_best import BEST_VAE_CONFIGS, train_vae_final

# This script's output (per-fold results CSV, stability summary CSV) and log
# all go here - created right at import time so a shell log redirect works
# from the very first run (see the same note in retrain_ae_9dim.py).
OUTPUT_DIR = Path("retrain_ae_5fold_cv")
OUTPUT_DIR.mkdir(exist_ok=True)

N_FOLDS = 5
RANDOM_SEED = 42
EVAL_PERCENTILE = 95

ARCHITECTURES = ["linear32_relu", "linear9_relu", "vae_best"]


def reconstruct(model, X_t):
    """Dispatch to model.reconstruct() for the VAE (deterministic decode from
    mu, no sampling) or a plain forward() pass for the Linear architectures."""
    if hasattr(model, "reconstruct"):
        return model.reconstruct(X_t)
    return model(X_t)


def train_model(arch, X_train, dataset_name):
    if arch == "linear32_relu":
        return train_ae32(X_train, epochs=30, batch_size=16, learning_rate=0.001)
    if arch == "linear9_relu":
        return construct_ae_9dim(X_train, epochs=30, batch_size=16, learning_rate=0.001)
    if arch == "vae_best":
        config = BEST_VAE_CONFIGS[dataset_name]
        input_dim = X_train.shape[1]
        return train_vae_final(
            X_train, input_dim,
            activation_name=config["activation"],
            beta=config["beta"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
        )
    raise ValueError(f"Unknown architecture: {arch}")


def evaluate_fold(model, X_train_normal_t, X_test_t, y_test, cus_percentile=EVAL_PERCENTILE):
    model.eval()
    with torch.no_grad():
        recon_train = reconstruct(model, X_train_normal_t)
        recon_error_train = torch.mean(torch.pow(X_train_normal_t - recon_train, 2), dim=1).numpy()
        threshold = np.percentile(recon_error_train, cus_percentile)

        recon_test = reconstruct(model, X_test_t)
        recon_error_test = torch.mean(torch.pow(X_test_t - recon_test, 2), dim=1).numpy()

    y_pred = (recon_error_test > threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "threshold": threshold,
        "precision_attack": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_attack": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1_attack": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1_weighted": report["weighted avg"]["f1-score"],
        "accuracy": report["accuracy"],
    }


def run_5fold_for_dataset(dataset_name, df):
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    df = df.copy()
    if "is_attack" not in df.columns:
        df["is_attack"] = (df["attack_specific"] != 0).astype(int)
        if "attack_specific" in df.columns:
            df = df.drop(columns=["attack_specific"])
    df = df.reset_index(drop=True)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_indices = list(kf.split(df))

    results = []
    dataset_start = time.time()

    for fold_i, (train_idx, test_idx) in enumerate(fold_indices, start=1):
        df_train_subsets = df.iloc[train_idx]
        df_test_subset = df.iloc[test_idx]

        # only normal rows from the other 4 subsets are used for training
        df_train_normal = df_train_subsets[df_train_subsets["is_attack"] == 0]
        X_train_normal = df_train_normal.drop(columns=["is_attack"])
        X_train_normal_t = torch.tensor(X_train_normal.values, dtype=torch.float32)

        # the ENTIRE held-out subset (normal + attack, natural proportion) is the test set
        X_test = df_test_subset.drop(columns=["is_attack"])
        y_test = df_test_subset["is_attack"].values
        X_test_t = torch.tensor(X_test.values, dtype=torch.float32)

        print(f"\n--- Fold {fold_i}/{N_FOLDS} (held-out subset{fold_i}) ---")
        print(f"Train (normal only, from the other {N_FOLDS - 1} subsets): {len(X_train_normal)} rows")
        print(f"Test (entire held-out subset{fold_i}): {len(X_test)} rows "
              f"({int((y_test == 0).sum())} normal, {int((y_test == 1).sum())} attack)")

        for arch in ARCHITECTURES:
            fold_start = time.time()
            model = train_model(arch, X_train_normal, dataset_name)
            metrics = evaluate_fold(model, X_train_normal_t, X_test_t, y_test)
            fold_time = time.time() - fold_start

            result = {
                "dataset": dataset_name,
                "architecture": arch,
                "fold": fold_i,
                "n_train_normal": len(X_train_normal),
                "n_test_total": len(X_test),
                "n_test_normal": int((y_test == 0).sum()),
                "n_test_attack": int((y_test == 1).sum()),
                "fold_time_s": fold_time,
                **metrics,
            }
            results.append(result)

            print(f"  [{arch}] f1_attack={metrics['f1_attack']:.4f}, "
                  f"precision={metrics['precision_attack']:.4f}, "
                  f"recall={metrics['recall_attack']:.4f}, time={fold_time:.2f}s")

    dataset_time = time.time() - dataset_start
    print(f"\nDataset {dataset_name} total time (all folds x all architectures): {dataset_time:.2f}s")

    results_df = pd.DataFrame(results)
    csv_name = dataset_name.lower().replace(" ", "_") + "_5fold_results.csv"
    results_df.to_csv(OUTPUT_DIR / csv_name, index=False)
    print(f"Saved per-fold results: {csv_name}")

    return results_df, dataset_time


def build_stability_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of f1/precision/recall across the 5 folds, per dataset x
    architecture. A low std means the model's performance doesn't depend much
    on which subset was held out (stable); a high std means it does."""
    rows = []
    for (dataset_name, arch), group in all_results.groupby(["dataset", "architecture"]):
        rows.append({
            "Dataset": dataset_name,
            "Architecture": arch,
            "F1 (attack) mean": round(group["f1_attack"].mean(), 4),
            "F1 (attack) std": round(group["f1_attack"].std(), 4),
            "Precision mean": round(group["precision_attack"].mean(), 4),
            "Precision std": round(group["precision_attack"].std(), 4),
            "Recall mean": round(group["recall_attack"].mean(), 4),
            "Recall std": round(group["recall_attack"].std(), 4),
            "F1 (weighted) mean": round(group["f1_weighted"].mean(), 4),
            "Accuracy mean": round(group["accuracy"].mean(), 4),
        })
    return pd.DataFrame(rows).sort_values(["Dataset", "Architecture"])


def main():
    ics_datasets = load_and_clean_datasets()

    all_results = []
    total_time = 0.0

    for dataset_name, df in ics_datasets.items():
        results_df, dataset_time = run_5fold_for_dataset(dataset_name, df)
        all_results.append(results_df)
        total_time += dataset_time

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(OUTPUT_DIR / "all_folds_results.csv", index=False)

    stability_summary = build_stability_summary(all_results_df)
    stability_summary.to_csv(OUTPUT_DIR / "5fold_stability_summary.csv", index=False)

    print("=" * 60)
    print("STABILITY SUMMARY (mean/std across 5 folds)")
    print("=" * 60)
    print(stability_summary.to_string(index=False))
    print("\nSaved: all_folds_results.csv, 5fold_stability_summary.csv")
    print(f"Total time consumed for all datasets: {total_time:.2f}s")


if __name__ == "__main__":
    main()
