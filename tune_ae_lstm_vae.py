"""
Tune and compare an nn.LSTM-based autoencoder and a Variational Autoencoder
(VAE) as alternatives to the plain Linear/MLP autoencoder tuned in
tune_ae_9dim.py.

For each dataset, runs a random hyperparameter search separately for the
LSTM autoencoder and the VAE, scores every trial by F1 on the attack
(anomaly) class (same methodology as tune_ae_9dim.py / retrain_ae_9dim.py),
and writes one combined results CSV per dataset with an "architecture"
column so the two can be compared directly (and compared against the
Linear baseline results from tune_ae_9dim.py's CSVs).

Note on the LSTM autoencoder: each packet here is a flat 18-feature vector
with no inherent temporal order between features, so the LSTM treats the
feature vector as a length-input_dim sequence of scalars (one feature per
timestep) rather than modeling true packet-to-packet sequence structure.

Run in a screen session, e.g.:
    screen -S ae_arch_tune
    python tune_ae_lstm_vae.py
    # Ctrl-A D to detach
"""

import time
import random
import itertools

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.metrics import f1_score, precision_score, recall_score

from retrain_ae_9dim import load_and_clean_datasets, preprocess_ae_9dim
from tune_ae_9dim import ACTIVATIONS

# ---------------------------------------------------------------------------
# Search space / tuning settings - adjust these to trade off runtime vs.
# search coverage. LSTM training is significantly slower per step than the
# Linear autoencoder, so defaults here are smaller than tune_ae_9dim.py.
# ---------------------------------------------------------------------------
LATENT_DIM = 9  # fixed, matches retrain_ae_9dim.py / tune_ae_9dim.py

LSTM_SEARCH_SPACE = {
    "epochs": [10, 20],
    "batch_size": [32, 64],
    "learning_rate": [0.001, 0.0005],
    "hidden_size": [16, 32],
}

VAE_SEARCH_SPACE = {
    "epochs": [10, 20],
    "batch_size": [32, 64],
    "learning_rate": [0.001, 0.0005],
    "activation": ["relu", "leaky_relu", "elu"],
    "beta": [0.1, 0.5, 1.0],   # weight of the KL-divergence term
}

N_TRIALS_PER_ARCH = 20      # number of random configs to try per architecture, per dataset
TRAIN_SAMPLE_SIZE = 3000    # subsample training rows per dataset for speed (None = use all)
EVAL_PERCENTILE = 95
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

# CLASS:    AutoEncoder_LSTM
# PURPOSE:  Autoencoder using nn.LSTM layers instead of plain nn.Linear.
class AutoEncoder_LSTM(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM, hidden_size=32):
        super().__init__()
        self.input_dim = input_dim

        self.encoder_lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_size, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.output_fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x_seq = x.unsqueeze(-1)  # (batch, input_dim, 1)
        _, (h_n, _) = self.encoder_lstm(x_seq)
        latent = self.encoder_fc(h_n[-1])  # (batch, latent_dim)

        dec_hidden = self.decoder_fc(latent)  # (batch, hidden_size)
        dec_input = dec_hidden.unsqueeze(1).repeat(1, self.input_dim, 1)  # (batch, input_dim, hidden_size)
        dec_out, _ = self.decoder_lstm(dec_input)
        return self.output_fc(dec_out).squeeze(-1)  # (batch, input_dim)

    def reconstruct(self, x):
        return self.forward(x)


# CLASS:    AutoEncoder_VAE
# PURPOSE:  Variational autoencoder: same Linear encoder/decoder shape as
#           AutoEncoder_Tunable, but the bottleneck is a sampled latent
#           (mu, logvar) instead of a deterministic one.
class AutoEncoder_VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM, activation=nn.ReLU):
        super().__init__()

        self.encoder_hidden = nn.Sequential(
            nn.Linear(input_dim, 128),
            activation(),
            nn.Linear(128, 64),
            activation(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            activation(),
            nn.Linear(64, 128),
            activation(),
            nn.Linear(128, input_dim),
        )

    def encode(self, x):
        h = self.encoder_hidden(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def reconstruct(self, x):
        # deterministic reconstruction for eval/anomaly scoring: use mu directly, no sampling
        mu, _ = self.encode(x)
        return self.decoder(mu)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_lstm(X_train, input_dim, hidden_size, epochs, batch_size, learning_rate):
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    model = AutoEncoder_LSTM(input_dim, LATENT_DIM, hidden_size)

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


def train_vae(X_train, input_dim, activation_name, beta, epochs, batch_size, learning_rate):
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    model = AutoEncoder_VAE(input_dim, LATENT_DIM, ACTIVATIONS[activation_name])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i in range(0, X_train_t.size(0), batch_size):
            batch = X_train_t[i:i + batch_size]
            optimizer.zero_grad()

            recon, mu, logvar = model(batch)
            recon_loss = nn.functional.mse_loss(recon, batch)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss

            loss.backward()
            optimizer.step()

    return model


# ---------------------------------------------------------------------------
# Evaluation - shared across architectures via model.reconstruct(x)
# ---------------------------------------------------------------------------
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
        recon_train = model.reconstruct(X_train_t)
        recon_error_train = torch.mean(torch.pow(X_train_t - recon_train, 2), dim=1).numpy()
        threshold = np.percentile(recon_error_train, cus_percentile)

        reconstructed = model.reconstruct(X_test_t)
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
    return [dict(zip(keys, combo)) for combo in all_combos[:n_trials]]


def tune_lstm(X_train, df_ae, input_dim):
    configs = sample_configs(LSTM_SEARCH_SPACE, N_TRIALS_PER_ARCH)
    results = []

    for i, config in enumerate(configs, start=1):
        trial_start = time.time()

        model = train_lstm(
            X_train, input_dim,
            hidden_size=config["hidden_size"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
        )
        metrics = evaluate_config(model, df_ae)

        trial_time = time.time() - trial_start
        result = {"architecture": "lstm", **config, **metrics, "trial_time_s": trial_time}
        results.append(result)

        print(f"[LSTM {i}/{len(configs)}] {config} -> "
              f"f1_attack={metrics['f1_attack']:.4f}, "
              f"precision_attack={metrics['precision_attack']:.4f}, "
              f"recall_attack={metrics['recall_attack']:.4f}, "
              f"time={trial_time:.2f}s")

    return results


def tune_vae(X_train, df_ae, input_dim):
    configs = sample_configs(VAE_SEARCH_SPACE, N_TRIALS_PER_ARCH)
    results = []

    for i, config in enumerate(configs, start=1):
        trial_start = time.time()

        model = train_vae(
            X_train, input_dim,
            activation_name=config["activation"],
            beta=config["beta"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
        )
        metrics = evaluate_config(model, df_ae)

        trial_time = time.time() - trial_start
        result = {"architecture": "vae", **config, **metrics, "trial_time_s": trial_time}
        results.append(result)

        print(f"[VAE {i}/{len(configs)}] {config} -> "
              f"f1_attack={metrics['f1_attack']:.4f}, "
              f"precision_attack={metrics['precision_attack']:.4f}, "
              f"recall_attack={metrics['recall_attack']:.4f}, "
              f"time={trial_time:.2f}s")

    return results


def tune_dataset(dataset_name, df):
    print("=========================================")
    print(f"Tuning architectures for dataset: {dataset_name}")

    df_ae = df.copy()
    X_train_full, _ = preprocess_ae_9dim(df_ae)
    input_dim = X_train_full.shape[1]

    if TRAIN_SAMPLE_SIZE is not None and len(X_train_full) > TRAIN_SAMPLE_SIZE:
        X_train = X_train_full.sample(n=TRAIN_SAMPLE_SIZE, random_state=RANDOM_SEED)
    else:
        X_train = X_train_full

    dataset_start = time.time()

    results = []
    results += tune_lstm(X_train, df_ae, input_dim)
    results += tune_vae(X_train, df_ae, input_dim)

    dataset_time = time.time() - dataset_start
    print(f"Dataset {dataset_name} architecture tuning total time: {dataset_time:.2f}s")

    results_df = pd.DataFrame(results).sort_values("f1_attack", ascending=False)

    csv_name = dataset_name.lower().replace(" ", "_") + "_ae_arch_tuning_results.csv"
    results_df.to_csv(csv_name, index=False)

    best_per_arch = {
        arch: results_df[results_df["architecture"] == arch].iloc[0].to_dict()
        for arch in results_df["architecture"].unique()
    }

    print(f"Best LSTM config for {dataset_name}: {best_per_arch.get('lstm')}")
    print(f"Best VAE config for {dataset_name}: {best_per_arch.get('vae')}")

    return dataset_time, best_per_arch


def main():
    ics_datasets = load_and_clean_datasets()

    total_time = 0.0
    best_overall = {}

    for dataset_name, df in ics_datasets.items():
        dataset_time, best_per_arch = tune_dataset(dataset_name, df)
        total_time += dataset_time
        best_overall[dataset_name] = best_per_arch

    print("=========================================")
    print("Best config per dataset & architecture:")
    for dataset_name, best_per_arch in best_overall.items():
        for arch, best in best_per_arch.items():
            print(f"  {dataset_name} / {arch}: f1_attack={best['f1_attack']:.4f}")
    print(f"Total time consumed for all datasets: {total_time:.2f}s")


if __name__ == "__main__":
    main()
