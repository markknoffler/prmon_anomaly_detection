import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.expanduser("../data/analysis")
FIG  = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, "combined_dataset.csv")).fillna(0.0)

# ── Compute z-score against GLOBAL normal distribution ────────────────────────
# Use μ and σ from normal runs only — this is the correct approach for
# cross-run anomaly detection. Per-run rolling z-score (the precomputed
# pss_zscore column) catches up to any level shift within ~5 timesteps,
# making it blind to sustained anomalies like mem_spike.
normal_pss = df[df["label"] == 0]["pss"]
mu  = normal_pss.mean()
std = normal_pss.std()

df["global_zscore"] = (df["pss"] - mu) / std

Z_THRESHOLD = 3.0
df["pred_zscore"] = (np.abs(df["global_zscore"]) > Z_THRESHOLD).astype(int)
df["flag_type"] = np.where(
    (df["label"]==1) & (df["pred_zscore"]==1), "TP",
    np.where((df["label"]==0) & (df["pred_zscore"]==1), "FP",
    np.where((df["label"]==1) & (df["pred_zscore"]==0), "FN", "TN")))

y      = df["label"].values
preds  = df["pred_zscore"].values
scores = np.abs(df["global_zscore"].values)

print(f"Normal PSS — μ={mu:.0f} kB  σ={std:.0f} kB")
print(f"3σ threshold = {mu + 3*std:.0f} kB  ({(mu + 3*std)/1024:.1f} MB)")
print(f"anomaly_mem_spike PSS = 1,173,051 kB  → z = {(1173051 - mu)/std:.1f}σ")

prec = precision_score(y, preds, zero_division=0)
rec  = recall_score(y, preds, zero_division=0)
f1   = f1_score(y, preds, zero_division=0)
auc  = roc_auc_score(y, scores)

print("="*52)
print(f"Z-Score Detector (global PSS, threshold={Z_THRESHOLD}σ)")
print("="*52)
print(classification_report(y, preds, target_names=["Normal", "Anomaly"]))
print(f"ROC-AUC: {auc:.4f}")
print("\nPer anomaly-type detection rate:")
for atype, g in df[df["label"]==1].groupby("anomaly_type"):
    print(f"  {atype:24s}: {g['pred_zscore'].mean()*100:.1f}%")

pd.DataFrame([{
    "model":     "Z-Score",
    "precision": round(prec, 4),
    "recall":    round(rec,  4),
    "f1":        round(f1,   4),
    "roc_auc":   round(auc,  4),
}]).to_csv(os.path.join(BASE, "metrics_zscore.csv"), index=False)

# Save with global_zscore replacing pss_zscore for plotting
df["pss_zscore"] = df["global_zscore"]   # ← keeps figures.py working unchanged
df[["run_id","wtime","pss","pss_zscore","label","pred_zscore",
    "anomaly_type","flag_type"]].to_csv(
    os.path.join(BASE, "results_zscore.csv"), index=False)

