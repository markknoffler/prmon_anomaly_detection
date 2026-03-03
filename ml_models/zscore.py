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

Z_THRESHOLD = 3.0
df["pred_zscore"] = (np.abs(df["pss_zscore"]) > Z_THRESHOLD).astype(int)
df["flag_type"]   = np.where(
    (df["label"]==1) & (df["pred_zscore"]==1), "TP",
    np.where((df["label"]==0) & (df["pred_zscore"]==1), "FP",
    np.where((df["label"]==1) & (df["pred_zscore"]==0), "FN", "TN")))

y = df["label"].values
preds  = df["pred_zscore"].values
scores = np.abs(df["pss_zscore"].values)

prec = precision_score(y, preds, zero_division=0)
rec  = recall_score(y, preds, zero_division=0)
f1   = f1_score(y, preds, zero_division=0)
auc  = roc_auc_score(y, scores)

print("="*52)
print(f"Z-Score Detector (PSS, threshold={Z_THRESHOLD}σ)")
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

df[["run_id","wtime","pss","pss_zscore","label","pred_zscore",
    "anomaly_type","flag_type"]].to_csv(
    os.path.join(BASE, "results_zscore.csv"), index=False)

ANOM_RUNS = [r for r in df["run_id"].unique() if df[df["run_id"]==r]["label"].iloc[0]==1]
fig, axes = plt.subplots(len(ANOM_RUNS), 2, figsize=(14, 3.5*len(ANOM_RUNS)), facecolor="white")
for row, rid in enumerate(ANOM_RUNS):
    g = df[df["run_id"]==rid].sort_values("wtime")
    ax_pss, ax_z = axes[row]
    ax_pss.plot(g["wtime"], g["pss"]/1024, "#EF4444", lw=1.8)
    ax_pss.fill_between(g["wtime"], g["pss"]/1024, alpha=0.1, color="#EF4444")
    ax_pss.set_title(rid, fontsize=10, fontweight="bold")
    ax_pss.set_ylabel("PSS (MB)"); ax_pss.set_xlabel("Wall-time (s)")
    ax_pss.grid(alpha=0.25)
    ax_z.plot(g["wtime"], np.abs(g["pss_zscore"]), "#2563EB", lw=1.8, label="|Z-score|")
    ax_z.axhline(Z_THRESHOLD, color="#DC2626", ls="--", lw=2, label=f"{Z_THRESHOLD}σ threshold")
    ax_z.set_ylabel("|Z-score|"); ax_z.set_xlabel("Wall-time (s)")
    ax_z.set_ylim(0, max(4.5, np.abs(g["pss_zscore"]).max()*1.1))
    ax_z.grid(alpha=0.25); ax_z.legend(fontsize=9)
plt.suptitle("Z-Score Detector: PSS and |Z-score| per Anomaly Run", fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(FIG, "04_zscore_flags.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results_zscore.csv  metrics_zscore.csv  04_zscore_flags.png")
