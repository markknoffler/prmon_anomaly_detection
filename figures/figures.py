import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import roc_curve, auc

BASE    = "../data/analysis"
DL_DIR  = os.path.join(BASE, "dl_results")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

iso    = pd.read_csv(os.path.join(BASE, "results_isolation_forest.csv"))
ocsvm  = pd.read_csv(os.path.join(BASE, "results_ocsvm.csv"))
zsc    = pd.read_csv(os.path.join(BASE, "results_zscore.csv"))
dl     = pd.read_csv(os.path.join(DL_DIR, "results_dl.csv"))
dl = dl.rename(columns={"pss_kb": "pss"})
hist   = pd.read_csv(os.path.join(DL_DIR, "training_history.csv"))

m_iso   = pd.read_csv(os.path.join(BASE, "metrics_isolation_forest.csv"))
m_ocsvm = pd.read_csv(os.path.join(BASE, "metrics_ocsvm.csv"))
m_zsc   = pd.read_csv(os.path.join(BASE, "metrics_zscore.csv"))
m_dl    = pd.read_csv(os.path.join(DL_DIR, "metrics_dl.csv"))
metrics = pd.concat([m_iso, m_ocsvm, m_zsc, m_dl], ignore_index=True)
print("Metrics loaded:\n", metrics[["model","precision","recall","f1","roc_auc"]].to_string(index=False))

ANOM_TYPES = ["mem_spike", "thread_spike", "io_burst", "combined"]
ANOM_RUNS  = [f"anomaly_{t}" for t in ANOM_TYPES]
ANOM_LABEL = {"anomaly_mem_spike":"Memory Spike","anomaly_thread_spike":"Thread Spike",
              "anomaly_io_burst":"I/O Burst","anomaly_combined":"Combined"}

FLAG_COLORS = {"TP":"#16A34A", "FN":"#DC2626", "FP":"#F97316", "TN":"#94A3B8"}
FLAG_MARKERS= {"TP":"o",       "FN":"x",       "FP":"^"}

fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
ax.plot(hist["epoch"], hist["train_loss"], lw=2, color="#2563EB", label="Train loss")
ax.plot(hist["epoch"], hist["val_loss"],   lw=2, color="#EF4444", ls="--", label="Val loss")
best_ep = int(hist["val_loss"].idxmin()) + 1
ax.axvline(best_ep, color="#64748B", ls=":", lw=1.5, label=f"Best epoch ({best_ep})")
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("MSE Loss", fontsize=11)
ax.set_title("TA-LSTM-AE Training Curve — Train and Val Track Together", fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_training_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("01_training_curve.png")

metric_cols   = ["precision", "recall", "f1", "roc_auc"]
metric_labels = ["Precision", "Recall", "F1", "ROC-AUC"]
model_colors  = {
    "Isolation Forest": "#2563EB",
    "One-Class SVM":    "#7C3AED",
    "Z-Score":          "#059669",
    "TA-LSTM-AE":       "#EF4444",
}
x = np.arange(len(metric_labels))
width = 0.18
fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
for i, (_, row) in enumerate(metrics.iterrows()):
    bars = ax.bar(x + i * width, [row[m] for m in metric_cols],
                  width=width, label=row["model"],
                  color=model_colors.get(row["model"], "#94A3B8"),
                  alpha=0.88)
    for bar, val in zip(bars, [row[m] for m in metric_cols]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_labels, fontsize=12)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.25)
ax.set_title("Detection Metrics — All Models", fontweight="bold", fontsize=13)
ax.legend(fontsize=10, loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_metrics_all_models.png", dpi=150, bbox_inches="tight")
plt.close()
print("02_metrics_all_models.png")

model_map = [
    ("Isolation Forest", iso,   "pred_iso"),
    ("One-Class SVM",    ocsvm, "pred_ocsvm"),
    ("Z-Score",          zsc,   "pred_zscore"),
    ("TA-LSTM-AE",       dl,    "pred"),
]
heat = []
for mname, df_, pcol in model_map:
    row_data = []
    for atype in ANOM_TYPES:
        pattern = f"anomaly_{atype}" if f"anomaly_{atype}" in df_["run_id"].values else atype
        g = df_[df_["run_id"] == pattern]
        row_data.append(round(g[pcol].mean(), 3) if len(g) else 0.0)
    heat.append(row_data)
heat = np.array(heat)

fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
im = ax.imshow(heat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Detection Rate")
ax.set_xticks(range(len(ANOM_TYPES)))
ax.set_xticklabels(["Mem Spike","Thread Spike","I/O Burst","Combined"], fontsize=11)
ax.set_yticks(range(len(model_map)))
ax.set_yticklabels([m[0] for m in model_map], fontsize=11)
for i in range(len(model_map)):
    for j in range(len(ANOM_TYPES)):
        ax.text(j, i, f"{heat[i,j]*100:.0f}%", ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="black" if 0.25 < heat[i,j] < 0.85 else "white")
ax.set_title("Detection Rate by Model and Anomaly Type", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_detection_rate_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("03_detection_rate_heatmap.png")

def plot_timeseries_flags(df_, pred_col, model_name, out_fname, accent_color):
    runs = [r for r in ANOM_RUNS if r in df_["run_id"].values]
    fig, axes = plt.subplots(len(runs), 1, figsize=(13, 4*len(runs)), facecolor="white")
    if len(runs) == 1:
        axes = [axes]
    for ax, rid in zip(axes, runs):
        g = df_[df_["run_id"]==rid].sort_values("wtime")
        pss_mb = g["pss"] / 1024

        # Full PSS trace
        ax.fill_between(g["wtime"], pss_mb, alpha=0.07, color=accent_color)
        ax.plot(g["wtime"], pss_mb, color="#475569", lw=2, label="PSS (MB)", zorder=2)

        # Flag each TP/FN/FP
        for ftype, marker, color, size in [
            ("TP", "o", "#16A34A", 50),
            ("FN", "x", "#DC2626", 60),
            ("FP", "^", "#F97316", 50),
        ]:
            sub = g[g["flag_type"]==ftype]
            if len(sub):
                ax.scatter(sub["wtime"], sub["pss"]/1024,
                           c=color, s=size, marker=marker,
                           zorder=5, linewidths=1.5 if marker=="x" else 0,
                           label=f"{ftype} ({len(sub)})")

        ax.set_title(f"{model_name} — {ANOM_LABEL.get(rid, rid)}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Wall-time (s)", fontsize=10)
        ax.set_ylabel("PSS (MB)", fontsize=10)
        ax.grid(alpha=0.25); ax.legend(fontsize=9, loc="upper right")

    legend_elements = [
        Line2D([0],[0], color="#475569", lw=2, label="PSS trace"),
        plt.scatter([],[],c="#16A34A",s=50,marker="o",label="TP — correctly flagged"),
        plt.scatter([],[],c="#DC2626",s=60,marker="x",label="FN — missed anomaly"),
        plt.scatter([],[],c="#F97316",s=50,marker="^",label="FP — false alarm"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.01), framealpha=0.95)
    plt.suptitle(f"{model_name}: Anomaly Flags on PSS Time Series",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(pad=2.5)
    plt.savefig(os.path.join(FIG_DIR, out_fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(out_fname)

plot_timeseries_flags(iso,   "pred_iso",    "Isolation Forest", "04_if_timeseries_flags.png",   "#2563EB")
plot_timeseries_flags(ocsvm, "pred_ocsvm",  "One-Class SVM",    "05_ocsvm_timeseries_flags.png","#7C3AED")
plot_timeseries_flags(zsc,   "pred_zscore", "Z-Score",          "06_zscore_timeseries_flags.png","#059669")

rid_mem = "anomaly_mem_spike"
if rid_mem in zsc["run_id"].values:
    g = zsc[zsc["run_id"]==rid_mem].sort_values("wtime")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), facecolor="white",
                                    sharex=True, gridspec_kw={"height_ratios":[0.55,0.45]})
    fig.suptitle("Z-Score Blindspot: Level-Shift Anomaly Goes Undetected", fontsize=13, fontweight="bold")
    ax1.plot(g["wtime"], g["pss"]/1024, "#EF4444", lw=2.5, label="PSS (MB)")
    ax1.fill_between(g["wtime"], g["pss"]/1024, alpha=0.12, color="#EF4444")
    ax1.set_ylabel("PSS (MB)", fontsize=11); ax1.grid(alpha=0.25); ax1.legend(fontsize=10)
    ax1.set_title("PSS spikes from ~500 MB to 1,146 MB within 2 seconds", fontsize=10)
    ax2.plot(g["wtime"], g["pss_zscore"].abs(), "#2563EB", lw=2, label="|Z-score|")
    ax2.axhline(3.0, color="#DC2626", ls="--", lw=2, label="3σ threshold")
    ax2.set_ylabel("|Z-score|", fontsize=11); ax2.set_xlabel("Wall-time (s)", fontsize=11)
    ax2.set_title("Z-score stays near 0 once PSS plateaus — never crosses 3σ", fontsize=10)
    ax2.set_ylim(0, 4.5); ax2.grid(alpha=0.25); ax2.legend(fontsize=10)
    plt.tight_layout(pad=2.0)
    plt.savefig(f"{FIG_DIR}/07_zscore_failure_demo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("07_zscore_failure_demo.png")

threshold = float(m_dl["threshold"].iloc[0])
fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")
RUN_COLORS = {"normal_io_01":"#64748B","normal_cpu_01":"#94A3B8",
              "anomaly_mem_spike":"#EF4444","anomaly_thread_spike":"#F97316",
              "anomaly_io_burst":"#8B5CF6","anomaly_combined":"#EC4899"}
offset, xticks, xlbls = 0, [], []
for rid, g in dl.groupby("run_id", sort=False):
    x = np.arange(offset, offset + len(g))
    ax.bar(x, g["recon_error"].values, color=RUN_COLORS.get(rid,"#94A3B8"), alpha=0.82, width=1.0)
    xticks.append(offset + len(g)//2)
    xlbls.append(rid.replace("_","\n"))
    offset += len(g) + 4
ax.axhline(threshold, color="black", ls="--", lw=2, label=f"Threshold = {threshold:.4f}")
ax.set_xticks(xticks); ax.set_xticklabels(xlbls, fontsize=8)
ax.set_ylabel("Reconstruction Error", fontsize=11)
ax.set_title("TA-LSTM-AE: Per-Sequence Reconstruction Error — Normal Stays Below Threshold",
             fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_dl_reconstruction_errors.png", dpi=150, bbox_inches="tight")
plt.close()
print("08_dl_reconstruction_errors.png")

plot_timeseries_flags(dl, "pred", "TA-LSTM-AE", "09_dl_timeseries_flags.png", "#EF4444")

from sklearn.metrics import confusion_matrix
model_results = [
    ("Isolation Forest", iso["label"],   iso["pred_iso"]),
    ("One-Class SVM",    ocsvm["label"], ocsvm["pred_ocsvm"]),
    ("Z-Score",          zsc["label"],   zsc["pred_zscore"]),
    ("TA-LSTM-AE",       dl["label"],    dl["pred"]),
]
fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor="white")
for ax, (mname, y_true, y_pred) in zip(axes.flat, model_results):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal","Anomaly"], yticklabels=["Normal","Anomaly"],
                annot_kws={"size":14})
    ax.set_title(mname, fontweight="bold", fontsize=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")
plt.tight_layout(pad=2.0)
plt.savefig(f"{FIG_DIR}/10_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("10_confusion_matrices.png")

roc_data = [
    ("Isolation Forest", iso["label"],   iso["score_iso"]),
    ("One-Class SVM",    ocsvm["label"], ocsvm["score_ocsvm"]),
    ("Z-Score",          zsc["label"],   zsc["pss_zscore"].abs()),
    ("TA-LSTM-AE",       dl["label"],    dl["recon_error"]),
]
roc_colors = ["#2563EB","#7C3AED","#059669","#EF4444"]
fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
for (mname, y_true, y_score), color in zip(roc_data, roc_colors):
    if len(set(y_true)) < 2:
        continue
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2.5, color=color, label=f"{mname}  AUC={roc_auc:.3f}")
ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — All Models", fontweight="bold", fontsize=13)
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/11_roc_curves_all_models.png", dpi=150, bbox_inches="tight")
plt.close()
print("11_roc_curves_all_models.png")

print(f"\nAll figures saved to {FIG_DIR}/")

