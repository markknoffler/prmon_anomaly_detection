import pandas as pd
import numpy as np
import os, glob

BASE = os.path.expanduser('/Users/samreedhbhuyan/Desktop/Win_C/CERN/PERMON/data')
OUT  = os.path.join(BASE, 'analysis')
os.makedirs(OUT, exist_ok=True)

BASELINE_FILES = {
    'normal_mem_01.txt':  ('baseline', 'mem_normal',       0),
    'normal_mem_02.txt':  ('baseline', 'mem_normal',       0),
    'normal_mem_03.txt':  ('baseline', 'mem_normal_light', 0),
    'normal_mem_04.txt':  ('baseline', 'mem_normal_heavy', 0),
    'normal_cpu_01.txt':  ('baseline', 'cpu_normal',       0),
    'normal_io_01.txt':   ('baseline', 'io_normal',        0),
}
ANOMALY_FILES = {
    'anomaly_mem_spike.txt':    ('anomalous', 'mem_spike',    1),
    'anomaly_thread_spike.txt': ('anomalous', 'thread_spike', 1),
    'anomaly_combined.txt':     ('anomalous', 'combined',     1),
    'anomaly_io_burst.txt':     ('anomalous', 'io_burst',     1),
}

def load_and_label(fname, subdir, atype, label):
    path = os.path.join(BASE, subdir, fname)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping.")
        return None
    df = pd.read_csv(path, sep='\t')
    run_id = fname.replace('.txt', '')
    df['run_id']       = run_id
    df['anomaly_type'] = atype
    df['label']        = label
    # Normalize wall time to start at 0 per run
    df['wtime'] = df['wtime'] - df['wtime'].min()
    return df

frames = []
for fname, (sub, atype, lbl) in {**BASELINE_FILES, **ANOMALY_FILES}.items():
    df = load_and_label(fname, sub, atype, lbl)
    if df is not None:
        frames.append(df)

raw = pd.concat(frames, ignore_index=True)
print(f"Raw combined shape: {raw.shape}")
print(raw['label'].value_counts())

# ── Feature Engineering ─────────────────────────────────────────────────────

# Rate of PSS change (memory growth rate per second)
raw['dpss_dt'] = (
    raw.groupby('run_id')['pss'].diff() /
    raw.groupby('run_id')['wtime'].diff().replace(0, np.nan)
)

# CPU efficiency: fraction of wall time spent doing useful CPU work
raw['cpu_eff'] = (raw['utime'] + raw['stime']) / raw['wtime'].clip(lower=1)

# System call overhead: what fraction of CPU time is kernel vs user
raw['stime_ratio'] = raw['stime'] / (raw['utime'] + raw['stime']).clip(lower=1)

# Per-process memory: isolates whether memory growth is process-proportional
raw['pss_per_proc'] = raw['pss'] / raw['nprocs'].clip(lower=1)

# I/O throughput rate
raw['io_rate'] = raw['wchar'] / raw['wtime'].clip(lower=1)

# Rolling statistics of PSS (window=5 snapshots per run)
raw['pss_roll_mean'] = raw.groupby('run_id')['pss'].transform(
    lambda x: x.rolling(5, min_periods=1).mean()
)
raw['pss_roll_std'] = raw.groupby('run_id')['pss'].transform(
    lambda x: x.rolling(5, min_periods=1).std().fillna(0)
)

# Z-score of PSS relative to its rolling stats (used for z-score detector)
raw['pss_zscore'] = (
    (raw['pss'] - raw['pss_roll_mean']) /
    raw['pss_roll_std'].clip(lower=1)
)

raw = raw.fillna(0)

# ── Final Feature Columns ────────────────────────────────────────────────────

FEATURE_COLS = [
    'pss', 'rss', 'nthreads', 'nprocs',
    'utime', 'stime',
    'rchar', 'wchar',
    'dpss_dt', 'cpu_eff', 'stime_ratio',
    'pss_per_proc', 'io_rate',
    'pss_roll_mean', 'pss_roll_std',
]

# ── Shuffle and Save ─────────────────────────────────────────────────────────
# IMPORTANT: shuffle at run level, not row level.
# Shuffling individual rows would break time-series order within each run.
# Instead we shuffle the order in which runs appear in the dataset,
# then within each run the rows remain time-ordered.

run_ids = raw['run_id'].unique().tolist()
rng = np.random.default_rng(seed=42)
rng.shuffle(run_ids)

shuffled_frames = [raw[raw['run_id'] == r] for r in run_ids]
dataset = pd.concat(shuffled_frames, ignore_index=True)

dataset.to_csv(os.path.join(OUT, 'combined_dataset.csv'), index=False)
print(f"\nFinal dataset shape: {dataset.shape}")
print(f"Feature columns: {FEATURE_COLS}")
print(f"Saved to {OUT}/combined_dataset.csv")

