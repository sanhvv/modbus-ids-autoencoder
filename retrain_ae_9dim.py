"""
Standalone script version of the "Test AutoEncoder with nn.Linear(64, 9)"
section of ics_simlab_sanh.ipynb.

Loads + cleans the 3 ICS-SimLab datasets, then trains (or loads) a smaller
autoencoder (latent dim 9 instead of 32) for each dataset and evaluates it.

Run in a screen session, e.g.:
    screen -S ae_retrain
    python retrain_ae_9dim.py
    # Ctrl-A D to detach
"""

import os
import time
import ipaddress  # noqa: F401 (kept for parity with notebook, unused)
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path("data").resolve()

DATASET_FILENAMES = {
    "Intelligent Electronic Device": "dataset_ied_packetv4.csv",
    "Smart Grid": "dataset_sg_packetv4.csv",
    "Water Bottle Factory": "dataset_wbf_packetv4.csv",
}


def find_dataset_csv(filename: str) -> str:
    matches = list(DATA_DIR.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Khong tim thay '{filename}' trong '{DATA_DIR}/'. "
            f"Chay cell tai dataset (Kaggle) o duoi truoc, hoac copy file csv thu cong vao '{DATA_DIR}/'."
        )
    return str(matches[0])


# FUNCTION: clean_dataset
# PURPOSE:  Performs dataset cleaning
def clean_dataset(df, multiclass=True):
    df_c = df.copy()

    # drop all protocols apart from MODBUS
    if "protocol" in df_c.columns:
        df_c = df_c[df_c['protocol'] == 'MODBUS']
    df_c = df_c.drop("protocol", axis=1)

    # drop the time column
    df_c = df_c.drop("time", axis=1)

    # set NaN rows in "tcp_analysis_ack_rtt" to the mean
    mean_rtt = df_c["tcp_analysis_ack_rtt"].mean()
    df_c["tcp_analysis_ack_rtt"] = df_c["tcp_analysis_ack_rtt"].fillna(mean_rtt)

    # convert all "N/A" fields to 0 (affect only the attack categoty columns)
    df_c = df_c.fillna(0)
    df_c = df_c.replace("N/A", 0)

    # limit the modbus_data to 16 hex digits
    df_c['modbus_data'] = df_c['modbus_data'].astype(str).apply(lambda x: x[:12])

    # convert all hex to int
    df_c['ip_id'] = df_c['ip_id'].apply(lambda x: int(str(x), 16))
    df_c['ip_checksum'] = df_c['ip_checksum'].apply(lambda x: int(str(x), 16))
    df_c['tcp_flags'] = df_c['tcp_flags'].apply(lambda x: int(str(x), 16))
    df_c['modbus_data'] = df_c['modbus_data'].apply(
        lambda x: int(str(x).replace("0x", ""), 16) if str(x).replace("0x", "") else None
    )

    # drop mac and ip (may encode later)
    df_c = df_c.drop("ip_src", axis=1)
    df_c = df_c.drop("ip_dst", axis=1)
    df_c = df_c.drop("ether_src_mac", axis=1)
    df_c = df_c.drop("ether_dst_mac", axis=1)

    # drop clearly redundant columns
    df_c = df_c.drop("ip_checksum", axis=1)
    df_c = df_c.drop("modbus_data", axis=1)   # WILL PROBABLY CHANGE

    # standardize selected columns (may change)
    std_scaler = StandardScaler()
    dontStand = ["attack_binary", "attack_obj", "attack_specific",
                 "modbus_func_code",
                 "ip_flags_df", "ip_flags_mf"]

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

    # convert classifcation columns to integers
    df_c['attack_binary'] = df_c['attack_binary'].astype(int)
    df_c['attack_obj'] = df_c['attack_obj'].astype(int)
    df_c['attack_specific'] = df_c['attack_specific'].astype(int)

    # drop any features that cause label leaking
    df_c = df_c.drop("tcp_stream", axis=1)
    df_c = df_c.drop("frame_time_relative", axis=1)

    # drop attack info (may change) due to label leaking
    if multiclass:
        df_c = df_c.drop("attack_binary", axis=1)
        df_c = df_c.drop("attack_obj", axis=1)
    else:
        df_c = df_c.drop("attack_specific", axis=1)
        df_c = df_c.drop("attack_obj", axis=1)

    # reduce DDOS instances to avoid anomaly class overfill
    df_attack8 = df_c[df_c["attack_specific"] == 8]
    df_others = df_c[df_c["attack_specific"] != 8]
    df_attack8_reduced = df_attack8.sample(frac=0.10, random_state=42)  # keep 10%, adjust as needed
    df_c = pd.concat([df_others, df_attack8_reduced])

    return df_c


def load_and_clean_datasets():
    df_ied = pd.read_csv(find_dataset_csv(DATASET_FILENAMES["Intelligent Electronic Device"]))
    df_sg = pd.read_csv(find_dataset_csv(DATASET_FILENAMES["Smart Grid"]))
    df_wbf = pd.read_csv(find_dataset_csv(DATASET_FILENAMES["Water Bottle Factory"]))

    ics_datasets_orig = {
        "Intelligent Electronic Device": df_ied,
        "Smart Grid": df_sg,
        "Water Bottle Factory": df_wbf,
    }

    ics_datasets = {}
    for dataset_name, df in ics_datasets_orig.items():
        ics_datasets[dataset_name] = clean_dataset(df)

    for dataset_name, df in ics_datasets.items():
        print("===================================")
        print(f"Dataset: {dataset_name}")
        print(f"Shape: {df.shape}")
        print(f"Column Names: {df.columns}")

    return ics_datasets


# CLASS:    AutoEncoder_Test (Modified)
# PURPOSE:  A vanila autoencoder neural network with a smaller latent space.
class AutoEncoder_Test(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder_Test, self).__init__()

        # encoder (9 dim latent space)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# FUNCTION: preprocess_ae_9dim
# PURPOSE:  A function to further preprocess data specifically for the
#           autoencoder. Returns training and test data (no need for prediction
#           data in an autoencoder).
def preprocess_ae_9dim(df):
    # convert the multiclass dataset in a single class
    if "is_attack" not in df.columns:
        df["is_attack"] = (df["attack_specific"] != 0).astype(int)

        # drop the attack_specfic column once created the binary column
        if "attack_specific" in df.columns:
            df.drop("attack_specific", axis=1, inplace=True)

    # train only on normal data
    X = df.drop(columns=["is_attack"])
    y = df["is_attack"]

    X_train = X[y == 0]
    X_test = X

    return X_train, X_test


# FUNCTION: construct_ae_9dim
# PURPOSE:  Builds an autoencoder model for anomaly classification.
def construct_ae_9dim(X_train, epochs=30, batch_size=16, learning_rate=0.001):
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)

    input_dim = X_train_t.shape[1]
    model = AutoEncoder_Test(input_dim)

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


# FUNCTION: detect_anomaly_9dim
# PURPOSE:  Given an autoencoder, predicts an anomaly. An anomaly is true if
#           it's reconstructed error falls outside the given percentile.
def detect_anomaly_9dim(model, X_train, X_test, y_true, dataset_name, cus_percentile=95, return_originals=False):
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_test)
        reconstructed_mse = torch.mean(torch.pow(X_test - reconstructed, 2), dim=1).numpy()

    recon_train = model(X_train)
    recon_error_train = torch.mean(torch.pow(X_train - recon_train, 2), dim=1).detach().numpy()
    best_threshold = np.percentile(recon_error_train, cus_percentile)
    print("Threshold:", best_threshold)

    anomaly_labels = (reconstructed_mse > best_threshold).astype(int)

    if dataset_name is not None:
        file_threshold_name = dataset_name.lower().replace(" ", "_") + "_threshold_test.txt"
        with open(file_threshold_name, "w") as f:
            f.write(str(best_threshold))

    if return_originals:
        original_anomalies = X_test[anomaly_labels == 1]
        print(len(original_anomalies))
        return anomaly_labels, original_anomalies

    return anomaly_labels


# FUNCTION: eval_ae_9dim
# PURPOSE:  Evaluates an autoencoder. Includes a class balancing step for
#           fair evaluation.
def eval_ae_9dim(df, ae_model, dataset_name=None):
    df_normal = df[df["is_attack"] == 0]
    df_attack = df[df["is_attack"] == 1]
    df_normal = df_normal.sample(n=len(df_attack), random_state=42)
    df_balanced = pd.concat([df_normal, df_attack])

    X = df_balanced.drop(columns=["is_attack"])
    y = df_balanced["is_attack"]

    X_train_ae = X[y == 0]
    X_test_ae = X

    X_train_t = torch.tensor(X_train_ae.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_ae.values, dtype=torch.float32)

    y_true = df_balanced["is_attack"]

    y_pred, anomalies = detect_anomaly_9dim(ae_model, X_train_t, X_test_t, y_true, dataset_name, return_originals=True)

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True)
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    print("Number of anomlies detected:", len(anomalies))

    return report


def main():
    ics_datasets = load_and_clean_datasets()

    build_ae_model_test = True   # convenient flag to determine if to train the model
    ae_models_test = {}

    dataset_times = {}
    total_time = 0.0

    if build_ae_model_test:
        for dataset_name, df in ics_datasets.items():
            print("=========================================")
            print(f"Dataset: {dataset_name}")

            dataset_start = time.time()

            df_ae_copy = df.copy()
            X_train_ae, X_test_ae = preprocess_ae_9dim(df_ae_copy)
            df_ae_copy = None

            ae_model_test = construct_ae_9dim(X_train_ae)
            ae_models_test[dataset_name] = ae_model_test

            file_model_name = dataset_name.lower().replace(" ", "_") + "_ae_model_9dim.pt"
            torch.save(ae_model_test.state_dict(), file_model_name)

            dataset_time = time.time() - dataset_start
            dataset_times[dataset_name] = dataset_time
            total_time += dataset_time
            print(f"Dataset {dataset_name} total time: {dataset_time:.2f}s")

    for dataset_name, ae_model_test in ae_models_test.items():
        print("=========================================")
        print(f"Model for dataset (Test): {dataset_name}")

        df_ae_copy = ics_datasets[dataset_name].copy()
        preprocess_ae_9dim(df_ae_copy)

        eval_ae_9dim(df_ae_copy, ae_model_test, dataset_name)

    print("=========================================")
    for dataset_name, dataset_time in dataset_times.items():
        print(f"{dataset_name}: {dataset_time:.2f}s")
    print(f"Total time consumed for all datasets: {total_time:.2f}s")


if __name__ == "__main__":
    main()
