import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
)

from model import TA_LSTM_AE
from loss import per_sequence_error
from dataset import build_loaders, FEATURE_COLS

OUT_DIR  = "../data/analysis/dl_results"
CSV_PATH = "../data/analysis/combined_dataset.csv"
FIG_DIR  = "../data/analysis/figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def load_model(device):
    ckpt  = torch.load(os.path.join(OUT_DIR, "best_model.pt"), map_location=device)
    hp    = ckpt["hparams"]
    model = TA_LSTM_AE(len(FEATURE_COLS), hp["hidden_dim"], hp["n_layers"], hp["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}")
    return model, hp


@torch.no_grad()
def score_sequences(model, seqs, device, bs=64):
    errs, attns = [], []
    T = torch.FloatTensor(seqs)
    for i in range(0, len(T), bs):
        b = T[i:i+bs].to(device)
        recon, attn = model(b)
        errs.append(per_sequence_error(b, recon).cpu().numpy())
        attns.append(attn.cpu().numpy())
    return np.concatenate(errs), np.concatenate(attns)


def flag_type(label, pred):
    if label == 1 and pred == 1: return "TP"
    if label == 0 and pred == 1: return "FP"
    if label == 1 and pred == 0: return "FN"
    return "TN"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, test_data, _ = build_loaders(CSV_PATH, batch_size=64)
    model, hp = load_model(device)

    # ── Threshold from validation (no test leakage) ───────────────────────────
    val_errs = []
    with torch.no_grad():
        for (b,) in val_loader:
            b = b.to(device)
            recon, _ = model(b)
            val_errs.extend(per_sequence_error(b, recon).cpu().numpy())
    threshold = float(np.percentile(val_errs, 95))
    print(f"Threshold (val 95th pct): {threshold:.6f}")

    # ── Score every test sequence ─────────────────────────────────────────────
    records = []
    attn_store = {}
    for rid, data in test_data.items():
        errs, attn = score_sequences(model, data["sequences"], device)
        attn_store[rid] = {"attn": attn, "label": data["label"]}
        for seq_idx, (e, wt, pss) in enumerate(zip(
                errs, data["seq_wtimes"], data["seq_pss"])):
            pred = int(e > threshold)
            records.append({
                "run_id":       rid,
                "seq_idx":      seq_idx,
                "wtime":        float(wt),
                "pss_kb":       float(pss),
                "label":        data["label"],
                "anomaly_type": data["anomaly_type"],
                "recon_error":  float(e),
                "pred":         pred,
                "flag_type":    flag_type(data["label"], pred),
            })

    df = pd.DataFrame(records)
    y, yh, ys = df["label"].values, df["pred"].values, df["recon_error"].values

    # ── Classification metrics → metrics CSV ─────────────────────────────────
    prec   = precision_score(y, yh, zero_division=0)
    rec    = recall_score(y, yh, zero_division=0)
    f1     = f1_score(y, yh, zero_division=0)
    roc    = roc_auc_score(y, ys) if len(set(y)) > 1 else float("nan")

    metrics = pd.DataFrame([{
        "model": "TA-LSTM-AE",
        "threshold": threshold,
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "roc_auc":   round(roc,  4),
    }])
    metrics.to_csv(os.path.join(OUT_DIR, "metrics_dl.csv"), index=False)

    print("\n" + "="*54)
    print("TA-LSTM-AE — Test Set Results")
    print("="*54)
    print(classification_report(y, yh, target_names=["Normal", "Anomaly"]))
    print(f"ROC-AUC : {roc:.4f}")
    print("\nPer anomaly-type detection rate:")
    for atype, g in df[df["label"]==1].groupby("anomaly_type"):
        print(f"  {atype:24s}: {g['pred'].mean()*100:.1f}%")

    # ── Save per-sequence results ─────────────────────────────────────────────
    df.to_csv(os.path.join(OUT_DIR, "results_dl.csv"), index=False)
    print(f"\nresults_dl.csv  → {OUT_DIR}/results_dl.csv")
    print(f"metrics_dl.csv  → {OUT_DIR}/metrics_dl.csv")

    # ── Plot 1: Training curve ────────────────────────────────────────────────
    hist_path = os.path.join(OUT_DIR, "training_history.csv")
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
        ax.plot(hist["epoch"], hist["train_loss"], lw=2, label="Train loss")
        ax.plot(hist["epoch"], hist["val_loss"],   lw=2, ls="--", label="Val loss")
        best_ep = hist["val_loss"].idxmin() + 1
        ax.axvline(best_ep, color="red", ls=":", lw=1.5, label=f"Best epoch {best_ep}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
        ax.set_title("TA-LSTM-AE Training Curve", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/07_dl_training_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Plot 2: Reconstruction error per run (bar) ────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")
    offset, xticks, xlbls = 0, [], []
    RUN_COLORS = {
        "normal_io_01": "#64748B", "normal_cpu_01": "#94A3B8",
        "anomaly_mem_spike": "#EF4444", "anomaly_thread_spike": "#F97316",
        "anomaly_io_burst": "#8B5CF6", "anomaly_combined": "#EC4899",
    }
    for rid, g in df.groupby("run_id", sort=False):
        errs_run = g["recon_error"].values
        x = np.arange(offset, offset + len(errs_run))
        ax.bar(x, errs_run, color=RUN_COLORS.get(rid, "#94A3B8"), alpha=0.8, width=1.0)
        xticks.append(offset + len(errs_run) // 2)
        xlbls.append(rid.replace("_", "\n"))
        offset += len(errs_run) + 4
    ax.axhline(threshold, color="black", ls="--", lw=1.5,
               label=f"Threshold = {threshold:.4f}")
    ax.set_xticks(xticks); ax.set_xticklabels(xlbls, fontsize=8)
    ax.set_ylabel("Reconstruction Error"); ax.legend()
    ax.set_title("TA-LSTM-AE: Per-Sequence Reconstruction Errors by Run", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/08_dl_anomaly_scores.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Anomaly flags on PSS time series (per anomaly run) ────────────
    anom_runs = [r for r in df["run_id"].unique() if df[df["run_id"]==r]["label"].iloc[0]==1]
    if anom_runs:
        fig, axes = plt.subplots(len(anom_runs), 1, figsize=(12, 4*len(anom_runs)),
                                 facecolor="white")
        if len(anom_runs) == 1:
            axes = [axes]
        for ax, rid in zip(axes, anom_runs):
            g = df[df["run_id"]==rid].sort_values("wtime")
            pss_mb = g["pss_kb"] / 1024
            ax.fill_between(g["wtime"], pss_mb, alpha=0.08, color="#EF4444")
            ax.plot(g["wtime"], pss_mb, color="#64748B", lw=1.8, label="PSS (MB)")
            tp = g[g["flag_type"]=="TP"]
            fn = g[g["flag_type"]=="FN"]
            fp = g[g["flag_type"]=="FP"]
            if len(tp): ax.scatter(tp["wtime"], tp["pss_kb"]/1024,
                                   c="#16A34A", s=40, zorder=5, label="TP")
            if len(fn): ax.scatter(fn["wtime"], fn["pss_kb"]/1024,
                                   c="#DC2626", s=40, marker="x", zorder=5, label="FN")
            if len(fp): ax.scatter(fp["wtime"], fp["pss_kb"]/1024,
                                   c="#F97316", s=40, marker="^", zorder=5, label="FP")
            ax.set_title(f"TA-LSTM-AE — {rid}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Wall-time (s)"); ax.set_ylabel("PSS (MB)")
            ax.grid(alpha=0.25); ax.legend(fontsize=9)
        plt.tight_layout(pad=2.0)
        plt.savefig(f"{FIG_DIR}/11_dl_timeseries_flags.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Plot 4: Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(y, yh)
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="white")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    ax.set_title("TA-LSTM-AE Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/09_dl_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 5: ROC curve ─────────────────────────────────────────────────────
    if len(set(y)) > 1:
        fpr, tpr, _ = roc_curve(y, ys)
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
        ax.plot(fpr, tpr, lw=2, color="#2563EB", label=f"AUC = {roc:.3f}")
        ax.plot([0,1],[0,1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — TA-LSTM-AE", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/10_dl_roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"All plots saved → {FIG_DIR}")


if __name__ == "__main__":
    main()
