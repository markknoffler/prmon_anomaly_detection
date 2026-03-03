import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLS = [
    "pss", "rss", "nthreads", "nprocs",
    "utime", "stime", "rchar", "wchar",
    "dpss_dt", "cpu_eff", "io_rate",
]

NORMAL_TRAIN_RUNS = [
    "normal_mem_01", "normal_mem_02", "normal_mem_03", "normal_mem_04",
]

TEST_RUNS = {
    "normal_io_01":         {"label": 0, "anomaly_type": "io_normal"},
    "normal_cpu_01":        {"label": 0, "anomaly_type": "cpu_normal"},
    "anomaly_mem_spike":    {"label": 1, "anomaly_type": "mem_spike"},
    "anomaly_thread_spike": {"label": 1, "anomaly_type": "thread_spike"},
    "anomaly_io_burst":     {"label": 1, "anomaly_type": "io_burst"},
    "anomaly_combined":     {"label": 1, "anomaly_type": "combined"},
}


def sliding_windows(arr, seq_len):
    n = len(arr)
    if n < seq_len:
        return np.empty((0, seq_len, arr.shape[1]), dtype=np.float32)
    return np.stack([arr[i:i + seq_len] for i in range(n - seq_len + 1)], axis=0)


def build_loaders(csv_path, seq_len=10, batch_size=32,
                  scaler_save_path=None, val_frac=0.2, seed=42):
    df = pd.read_csv(csv_path)
    df = df.fillna(0.0)

    train_mask = df["run_id"].isin(NORMAL_TRAIN_RUNS)
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, FEATURE_COLS].values)
    if scaler_save_path:
        joblib.dump(scaler, scaler_save_path)

    all_seqs = []
    for rid in NORMAL_TRAIN_RUNS:
        rdf = df[df["run_id"] == rid].sort_values("wtime")
        scaled = scaler.transform(rdf[FEATURE_COLS].values).astype(np.float32)
        seqs = sliding_windows(scaled, seq_len)
        all_seqs.append(seqs)

    all_seqs = np.concatenate(all_seqs, axis=0)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_seqs))
    split = int(len(idx) * (1 - val_frac))
    tr_idx, va_idx = idx[:split], idx[split:]
    tr_seqs, va_seqs = all_seqs[tr_idx], all_seqs[va_idx]

    tr_loader = DataLoader(TensorDataset(torch.FloatTensor(tr_seqs)),
                           batch_size=batch_size, shuffle=True,  drop_last=False)
    va_loader = DataLoader(TensorDataset(torch.FloatTensor(va_seqs)),
                           batch_size=batch_size, shuffle=False, drop_last=False)

    test_data = {}
    for rid, meta in TEST_RUNS.items():
        rdf = df[df["run_id"] == rid].sort_values("wtime").reset_index(drop=True)
        if len(rdf) == 0:
            continue
        scaled = scaler.transform(rdf[FEATURE_COLS].values).astype(np.float32)
        seqs = sliding_windows(scaled, seq_len)
        if len(seqs) == 0:
            continue
        wtimes = rdf["wtime"].values
        seq_wtimes = [wtimes[i + seq_len - 1] for i in range(len(seqs))]
        pss_vals = rdf["pss"].values
        seq_pss = [pss_vals[i + seq_len - 1] for i in range(len(seqs))]
        test_data[rid] = {
            "sequences": seqs,
            "label": meta["label"],
            "anomaly_type": meta["anomaly_type"],
            "seq_wtimes": np.array(seq_wtimes, dtype=np.float32),
            "seq_pss": np.array(seq_pss, dtype=np.float32),
        }

    return tr_loader, va_loader, test_data, scaler
