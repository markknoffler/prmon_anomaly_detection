import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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

FEATURE_COLS = [
    "pss", "rss", "nthreads", "nprocs",
    "utime", "stime", "rchar", "wchar",
    "dpss_dt", "cpu_eff", "io_rate",
]

X = df[FEATURE_COLS].values
y = df["label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(
    n_estimators=300,
    contamination=0.25,
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)
iso.fit(X_scaled)

preds  = (iso.predict(X_scaled) == -1).astype(int)
scores = -iso.score_samples(X_scaled)

df["pred_iso"]  = preds
df["score_iso"] = scores
df["flag_type"] = np.where(
    (df["label"]==1) & (df["pred_iso"]==1), "TP",
    np.where((df["label"]==0) & (df["pred_iso"]==1), "FP",
    np.where((df["label"]==1) & (df["pred_iso"]==0), "FN", "TN")))

prec = precision_score(y, preds, zero_division=0)
rec  = recall_score(y, preds, zero_division=0)
f1   = f1_score(y, preds, zero_division=0)
auc  = roc_auc_score(y, scores)

print("="*52)
print("Isolation Forest")
print("="*52)
print(classification_report(y, preds, target_names=["Normal", "Anomaly"]))
print(f"ROC-AUC: {auc:.4f}")
print("\nPer anomaly-type detection rate:")
for atype, g in df[df["label"]==1].groupby("anomaly_type"):
    print(f"  {atype:24s}: {g['pred_iso'].mean()*100:.1f}%")

pd.DataFrame([{
    "model": "Isolation Forest",
    "precision": round(prec, 4),
    "recall":    round(rec,  4),
    "f1":        round(f1,   4),
    "roc_auc":   round(auc,  4),
}]).to_csv(os.path.join(BASE, "metrics_isolation_forest.csv"), index=False)

df[["run_id","wtime","pss","label","pred_iso","score_iso",
    "anomaly_type","flag_type"]].to_csv(
    os.path.join(BASE, "results_isolation_forest.csv"), index=False)

ANOM_RUNS = [r for r in df["run_id"].unique() if df[df["run_id"]==r]["label"].iloc[0]==1]
fig, axes = plt.subplots(len(ANOM_RUNS), 1, figsize=(12, 4*len(ANOM_RUNS)),
                         facecolor="white")
if len(ANOM_RUNS) == 1:
    axes = [axes]
for ax, rid in zip(axes, ANOM_RUNS):
    g = df[df["run_id"]==rid].sort_values("wtime")
    pss_mb = g["pss"] / 1024
    ax.fill_between(g["wtime"], pss_mb, alpha=0.07, color="#EF4444")
    ax.plot(g["wtime"], pss_mb, color="#64748B", lw=1.8, label="PSS (MB)")
    tp = g[g["flag_type"]=="TP"]
    fn = g[g["flag_type"]=="FN"]
    if len(tp): ax.scatter(tp["wtime"], tp["pss"]/1024, c="#16A34A", s=40, zorder=5, label="TP")
    if len(fn): ax.scatter(fn["wtime"], fn["pss"]/1024, c="#DC2626", s=40,
                           marker="x", linewidths=1.5, zorder=5, label="FN (missed)")
    ax.set_title(f"Isolation Forest — {rid}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Wall-time (s)"); ax.set_ylabel("PSS (MB)")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(FIG, "05_isolation_forest_flags.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results_isolation_forest.csv  metrics_isolation_forest.csv  05_isolation_forest_flags.png")
