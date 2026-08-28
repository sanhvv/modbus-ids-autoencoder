"""
Hyperparameter + activation-function tuning for the 9-dim-latent autoencoder
IDS model (see retrain_ae_9dim.py).

Runs a random search over epochs / batch_size / learning_rate / activation
function for each dataset, on a subsample of the training data for speed,
and reports the best config per dataset by F1 score on the attack
(anomaly) class. Full results per trial are written to a CSV per dataset.

Output CSVs go into tune_ae_9dim/. Run in a screen session, redirecting the
log into the same folder, e.g.:
    screen -S ae_tune
    python tune_ae_9dim.py 2>&1 | tee tune_ae_9dim/run_$(date +%Y%m%d_%H%M).log
    # Ctrl-A D to detach
"""

import time
import random
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.metrics import f1_score, precision_score, recall_score

from retrain_ae_9dim import load_and_clean_datasets, preprocess_ae_9dim

# Ket qua CSV cua script nay gom vao day (xem ghi chu tuong tu trong
# retrain_ae_9dim.py - tao ngay luc import de shell redirect vao day hoat
# dong duoc tu dau).
OUTPUT_DIR = Path("tune_ae_9dim")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Search space / tuning settings - adjust these to trade off runtime vs.
# search coverage.
# ---------------------------------------------------------------------------
LATENT_DIM = 9  # fixed, matches retrain_ae_9dim.py

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}

SEARCH_SPACE = {
    "epochs": [10, 20],
    "batch_size": [32, 64, 128],
    "learning_rate": [0.01, 0.001, 0.0005],
    "activation": list(ACTIVATIONS.keys()),
}

N_TRIALS = 12               # number of random configs to try per dataset
TRAIN_SAMPLE_SIZE = 5000    # subsample training rows per dataset for speed (None = use all)
EVAL_PERCENTILE = 95
RANDOM_SEED = 42


# CLASS:    AutoEncoder_Tunable
# PURPOSE:  Same shape as AutoEncoder_Test (retrain_ae_9dim.py) but with a
#           configurable latent dim and activation function.
class AutoEncoder_Tunable(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM, activation=nn.ReLU):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            activation(),
            nn.Linear(128, 64),
            activation(),
            nn.Linear(64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            activation(),
            nn.Linear(64, 128),
            activation(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_config(X_train, input_dim, activation_name, epochs, batch_size, learning_rate):
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    model = AutoEncoder_Tunable(input_dim, LATENT_DIM, ACTIVATIONS[activation_name])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i in range(0, X_train_t.size(0), batch_size):
            batch = X_train_t[i:i + batch_size]
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()

    return model


def evaluate_config(model, df, cus_percentile=EVAL_PERCENTILE):
    df_normal = df[df["is_attack"] == 0]
    df_attack = df[df["is_attack"] == 1]
    df_normal = df_normal.sample(n=len(df_attack), random_state=RANDOM_SEED)
    df_balanced = pd.concat([df_normal, df_attack])

    X = df_balanced.drop(columns=["is_attack"])
    y_true = df_balanced["is_attack"]

    X_train_ae = X[y_true == 0]
    X_train_t = torch.tensor(X_train_ae.values, dtype=torch.float32)
    X_test_t = torch.tensor(X.values, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        recon_train = model(X_train_t)
        recon_error_train = torch.mean(torch.pow(X_train_t - recon_train, 2), dim=1).numpy()
        threshold = np.percentile(recon_error_train, cus_percentile)

        reconstructed = model(X_test_t)
        recon_error = torch.mean(torch.pow(X_test_t - reconstructed, 2), dim=1).numpy()

    y_pred = (recon_error > threshold).astype(int)

    return {
        "f1_attack": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_attack": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_attack": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def sample_configs(search_space, n_trials, seed=RANDOM_SEED):
    keys = list(search_space.keys())
    all_combos = list(itertools.product(*[search_space[k] for k in keys]))
    rng = random.Random(seed)
    rng.shuffle(all_combos)
    chosen = all_combos[:n_trials]
    return [dict(zip(keys, combo)) for combo in chosen]


def tune_dataset(dataset_name, df):
    print("=========================================")
    print(f"Tuning dataset: {dataset_name}")

    df_ae = df.copy()
    X_train_full, _ = preprocess_ae_9dim(df_ae)
    input_dim = X_train_full.shape[1]

    if TRAIN_SAMPLE_SIZE is not None and len(X_train_full) > TRAIN_SAMPLE_SIZE:
        X_train = X_train_full.sample(n=TRAIN_SAMPLE_SIZE, random_state=RANDOM_SEED)
    else:
        X_train = X_train_full

    configs = sample_configs(SEARCH_SPACE, N_TRIALS)

    results = []
    dataset_start = time.time()

    for i, config in enumerate(configs, start=1):
        trial_start = time.time()

        model = train_config(
            X_train, input_dim,
            activation_name=config["activation"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
        )

        metrics = evaluate_config(model, df_ae)

        trial_time = time.time() - trial_start

        result = {**config, **metrics, "trial_time_s": trial_time}
        results.append(result)

        print(f"[{i}/{len(configs)}] {config} -> "
              f"f1_attack={metrics['f1_attack']:.4f}, "
              f"precision_attack={metrics['precision_attack']:.4f}, "
              f"recall_attack={metrics['recall_attack']:.4f}, "
              f"time={trial_time:.2f}s")

    dataset_time = time.time() - dataset_start
    print(f"Dataset {dataset_name} tuning total time: {dataset_time:.2f}s")

    results_df = pd.DataFrame(results).sort_values("f1_attack", ascending=False)

    csv_name = dataset_name.lower().replace(" ", "_") + "_ae_tuning_results.csv"
    results_df.to_csv(OUTPUT_DIR / csv_name, index=False)

    best = results_df.iloc[0].to_dict()
    print(f"Best config for {dataset_name}: {best}")

    return results_df, dataset_time


def main():
    ics_datasets = load_and_clean_datasets()

    total_time = 0.0
    best_per_dataset = {}

    for dataset_name, df in ics_datasets.items():
        results_df, dataset_time = tune_dataset(dataset_name, df)
        total_time += dataset_time
        best_per_dataset[dataset_name] = results_df.iloc[0].to_dict()

    print("=========================================")
    print("Best configs:")
    for dataset_name, best in best_per_dataset.items():
        print(f"  {dataset_name}: {best}")
    print(f"Total time consumed for all datasets: {total_time:.2f}s")


if __name__ == "__main__":
    main()
