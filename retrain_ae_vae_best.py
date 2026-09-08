"""
Final training script for the VAE autoencoder architecture, using the best
hyperparameters found per dataset by tune_ae_lstm_vae.py (3rd attempt, narrowed
beta search space). See tune_ae_lstm_vae/tune_ae_lstm_vae_3rd_attempt_summary.csv
and tune_ae_lstm_vae/*_ae_arch_tuning_results_3rd_attempt.csv for the full
search results this script's BEST_VAE_CONFIGS were picked from.

Difference from the tuning run: tune_ae_lstm_vae.py trains each trial on only
a TRAIN_SAMPLE_SIZE=3000-row subsample of normal data for speed. This script
retrains the winning config for each dataset on the FULL normal training set
(same convention as retrain_ae_9dim.py), so the metrics reported here will
differ somewhat from the tuning run's numbers.

Reuses the VAE architecture (AutoEncoder_VAE) and activation map from
tune_ae_lstm_vae.py / tune_ae_9dim.py, and the dataset loading/cleaning and
9-dim-AE-style preprocessing from retrain_ae_9dim.py, so all 3 scripts stay in
sync on data handling.

Output (model .pt, threshold .txt) goes into retrain_ae_vae_best/. Run in a
screen session, redirecting the log into the same folder, e.g.:
    screen -S ae_vae_best
    python retrain_ae_vae_best.py 2>&1 | tee retrain_ae_vae_best/run_$(date +%Y%m%d_%H%M).log
    # Ctrl-A D to detach
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix

from retrain_ae_9dim import load_and_clean_datasets, preprocess_ae_9dim
from tune_ae_9dim import ACTIVATIONS
from tune_ae_lstm_vae import AutoEncoder_VAE, LATENT_DIM

# This script's output (model .pt, threshold .txt) and log all go here -
# created right at import time so a shell log redirect works from the very
# first run (see the same note in retrain_ae_9dim.py).
OUTPUT_DIR = Path("retrain_ae_vae_best")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
EVAL_PERCENTILE = 95

# Best VAE config per dataset, picked by highest f1_attack from
# tune_ae_lstm_vae's 3rd-attempt tuning run (narrowed beta in [0.05, 0.1, 0.2]).
# latent_dim is fixed at LATENT_DIM (imported from tune_ae_lstm_vae.py) for all
# datasets, matching the tuning search space.
BEST_VAE_CONFIGS = {
    "Intelligent Electronic Device": {
        "epochs": 20, "batch_size": 64, "learning_rate": 0.001,
        "activation": "elu", "beta": 0.2,
    },
    "Smart Grid": {
        "epochs": 20, "batch_size": 64, "learning_rate": 0.001,
        "activation": "elu", "beta": 0.2,
    },
    "Water Bottle Factory": {
        "epochs": 20, "batch_size": 64, "learning_rate": 0.001,
        "activation": "relu", "beta": 0.2,
    },
}


# FUNCTION: train_vae_final
# PURPOSE:  Trains AutoEncoder_VAE on the full normal training set for one
#           dataset, with per-epoch loss/time logging (matching the logging
#           convention in retrain_ae_9dim.py's construct_ae_9dim()).
def train_vae_final(X_train, input_dim, activation_name, beta, epochs, batch_size, learning_rate):
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    model = AutoEncoder_VAE(input_dim, LATENT_DIM, ACTIVATIONS[activation_name])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0

        for i in range(0, X_train_t.size(0), batch_size):
            batch = X_train_t[i:i + batch_size]

            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            recon_loss = nn.functional.mse_loss(recon, batch)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(X_train_t):.6f}, Time: {epoch_time:.2f}s")

    return model


# FUNCTION: evaluate_vae
# PURPOSE:  Evaluates a trained VAE using the same balanced-eval / percentile
#           threshold methodology as eval_ae_9dim()/detect_anomaly_9dim() in
#           retrain_ae_9dim.py, so results are directly comparable. Uses
#           model.reconstruct(x) (deterministic decode from mu, no sampling)
#           instead of model(x), since AutoEncoder_VAE.forward() returns a
#           (recon, mu, logvar) tuple.
def evaluate_vae(df, model, dataset_name, cus_percentile=EVAL_PERCENTILE):
    df_normal = df[df["is_attack"] == 0]
    df_attack = df[df["is_attack"] == 1]
    df_normal_bal = df_normal.sample(n=len(df_attack), random_state=RANDOM_SEED)
    df_balanced = pd.concat([df_normal_bal, df_attack])

    X = df_balanced.drop(columns=["is_attack"])
    y_true = df_balanced["is_attack"]

    X_train_ae = X[y_true == 0]
    X_train_t = torch.tensor(X_train_ae.values, dtype=torch.float32)
    X_test_t = torch.tensor(X.values, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        recon_train = model.reconstruct(X_train_t)
        recon_error_train = torch.mean(torch.pow(X_train_t - recon_train, 2), dim=1).numpy()
        threshold = np.percentile(recon_error_train, cus_percentile)

        reconstructed = model.reconstruct(X_test_t)
        recon_error = torch.mean(torch.pow(X_test_t - reconstructed, 2), dim=1).numpy()

    print("Threshold:", threshold)
    y_pred = (recon_error > threshold).astype(int)

    if dataset_name is not None:
        file_threshold_name = dataset_name.lower().replace(" ", "_") + "_threshold_vae.txt"
        with open(OUTPUT_DIR / file_threshold_name, "w") as f:
            f.write(str(threshold))

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True)
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    return report, threshold


def main():
    ics_datasets = load_and_clean_datasets()

    vae_models = {}
    dataset_times = {}
    total_time = 0.0

    for dataset_name, df in ics_datasets.items():
        print("=========================================")
        print(f"Dataset: {dataset_name}")

        config = BEST_VAE_CONFIGS[dataset_name]
        print(f"Best VAE config: {config}")

        dataset_start = time.time()

        df_ae_copy = df.copy()
        X_train_ae, _ = preprocess_ae_9dim(df_ae_copy)
        input_dim = X_train_ae.shape[1]

        model = train_vae_final(
            X_train_ae, input_dim,
            activation_name=config["activation"],
            beta=config["beta"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
        )
        vae_models[dataset_name] = model

        file_model_name = dataset_name.lower().replace(" ", "_") + "_ae_vae_model.pt"
        torch.save(model.state_dict(), OUTPUT_DIR / file_model_name)

        dataset_time = time.time() - dataset_start
        dataset_times[dataset_name] = dataset_time
        total_time += dataset_time
        print(f"Dataset {dataset_name} total time: {dataset_time:.2f}s")

    for dataset_name, model in vae_models.items():
        print("=========================================")
        print(f"Model for dataset (VAE): {dataset_name}")

        df_ae_copy = ics_datasets[dataset_name].copy()
        preprocess_ae_9dim(df_ae_copy)  # ensures "is_attack" column exists

        evaluate_vae(df_ae_copy, model, dataset_name)

    print("=========================================")
    for dataset_name, dataset_time in dataset_times.items():
        print(f"{dataset_name}: {dataset_time:.2f}s")
    print(f"Total time consumed for all datasets: {total_time:.2f}s")


if __name__ == "__main__":
    main()
