"""
Data loading, run-level splitting, and sequence generation.

Split (run-level to preserve temporal ordering):
  Train : normal_mem_01, normal_mem_02, normal_mem_03, normal_mem_04
  Val   : normal_cpu_01  (different behaviour - tests normal generalisation)
  Test  : normal_io_01 + all 4 anomaly runs  (held out entirely)

Why train only on normal?
  ATLAS production jobs have no anomaly labels. The autoencoder learns normal
  patterns from clean historical data; anything it cannot reconstruct is flagged.
  Labels are used ONLY for evaluating detection quality, never for training.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLS = [
    "pss", "rss", "nthreads",
    "utime", "stime",
    "rchar", "wchar",
    "dpss_dt", "cpu_eff",
    "pss_per_proc", "io_rate",
]

TRAIN_RUNS = ["normal_mem_01", "normal_mem_02", "normal_mem_03", "normal_mem_04"]
VAL_RUNS   = ["normal_cpu_01"]
TEST_RUNS  = ["normal_io_01",
              "anomaly_mem_spike", "anomaly_thread_spike",
              "anomaly_combined",  "anomaly_io_burst"]


def sliding_window(arr, seq_len):
    n = len(arr)
    if n < seq_len:
        return np.empty((0, seq_len, arr.shape[1]))
    return np.stack([arr[i:i + seq_len] for i in range(n - seq_len + 1)])


class PrmonSeqDataset(Dataset):
    def __init__(self, sequences):
        self.X = torch.FloatTensor(sequences)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx]


def build_loaders(csv_path, seq_len=10, batch_size=32, scaler_save_path=None):
    df = pd.read_csv(csv_path).fillna(0)

    scaler = StandardScaler()
    train_df = df[df["run_id"].isin(TRAIN_RUNS)]
    scaler.fit(train_df[FEATURE_COLS].values)
    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)

    def runs_to_sequences(run_ids, tag):
        parts = []
        for rid in run_ids:
            sub = df[df["run_id"] == rid].sort_values("wtime")
            scaled = scaler.transform(sub[FEATURE_COLS].values)
            seqs   = sliding_window(scaled, seq_len)
            print(f"  {tag} | {rid}: {len(sub)} rows -> {len(seqs)} sequences")
            if len(seqs) > 0:
                parts.append(seqs)
        return np.concatenate(parts) if parts else np.empty((0, seq_len, len(FEATURE_COLS)))

    print("\n[TRAIN]")
    tr = runs_to_sequences(TRAIN_RUNS, "TRAIN")
    print("[VAL]")
    va = runs_to_sequences(VAL_RUNS, "VAL")

    print("[TEST]")
    test_data = {}
    for rid in TEST_RUNS:
        sub = df[df["run_id"] == rid].sort_values("wtime")
        if len(sub) == 0:
            continue
        scaled = scaler.transform(sub[FEATURE_COLS].values)
        seqs   = sliding_window(scaled, seq_len)
        print(f"  TEST | {rid}: {len(sub)} rows -> {len(seqs)} sequences")
        if len(seqs) > 0:
            test_data[rid] = {
                "sequences"   : seqs,
                "label"       : int(sub["label"].iloc[0]),
                "anomaly_type": sub["anomaly_type"].iloc[0],
            }

    tr_loader = DataLoader(PrmonSeqDataset(tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(PrmonSeqDataset(va), batch_size=batch_size, shuffle=False)
    return tr_loader, va_loader, test_data, scaler
