"""
Evaluation for TA-LSTM-AE.
Threshold = 95th percentile of validation reconstruction errors.
Outputs classification report, 4 plots, and results_dl.csv.
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

from model   import TA_LSTM_AE
from loss    import per_sequence_error
from dataset import build_loaders, FEATURE_COLS

OUT_DIR  = "../data/analysis/dl_results"
CSV_PATH = "../data/analysis/combined_dataset.csv"
FIG_DIR  = "../data/analysis/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def load_model(device):
    ckpt  = torch.load(os.path.join(OUT_DIR, "best_model.pt"), map_location=device)
    hp    = ckpt["hparams"]
    model = TA_LSTM_AE(len(FEATURE_COLS), hp["hidden_dim"], hp["n_layers"], hp["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, test_data, _ = build_loaders(CSV_PATH, batch_size=64)
    model, hp = load_model(device)

    # Threshold from validation (normal data only -- no test leakage)
    val_errs = []
    with torch.no_grad():
        for b in val_loader:
            b = b.to(device)
            recon, _ = model(b)
            val_errs.extend(per_sequence_error(b, recon).cpu().numpy())
    threshold = np.percentile(val_errs, 95)
    print(f"Threshold (val 95th pct): {threshold:.6f}")

    # Score test runs
    records, attn_store = [], {}
    for rid, data in test_data.items():
        errs, attn = score_sequences(model, data["sequences"], device)
        attn_store[rid] = (attn, data["label"])
        for e in errs:
            records.append({"run_id": rid, "label": data["label"],
                            "anomaly_type": data["anomaly_type"],
                            "recon_error": e, "pred": int(e > threshold)})

    df = pd.DataFrame(records)
    y, yh, ys = df["label"].values, df["pred"].values, df["recon_error"].values

    print("\n" + "="*52)
    print("TA-LSTM-AE  |  Test Set Results")
    print("="*52)
    print(classification_report(y, yh, target_names=["Normal","Anomaly"]))
    if len(set(y)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y, ys):.4f}")

    print("\nPer anomaly type detection rate:")
    for atype, g in df[df["label"]==1].groupby("anomaly_type"):
        print(f"  {atype:22s}: {g['pred'].mean()*100:.1f}%")

    df.to_csv(os.path.join(OUT_DIR, "results_dl.csv"), index=False)

    # ── Plot 1: Training curve ────────────────────────────────────────────
    hist = pd.read_csv(os.path.join(OUT_DIR, "training_history.csv"))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist["epoch"], hist["train_loss"], label="Train", lw=2)
    ax.plot(hist["epoch"], hist["val_loss"],   label="Val",   lw=2, ls="--")
    ax.axvline(hist["val_loss"].idxmin()+1, color="red", ls=":", label="Best epoch")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Seq-Norm MSE")
    ax.set_title("TA-LSTM-AE Training Curve", fontweight="bold"); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/07_dl_training_curve.png", dpi=150); plt.close()

    # ── Plot 2: Anomaly scores per run ────────────────────────────────────
    cmap = {"normal_io_01":"steelblue", "anomaly_mem_spike":"crimson",
            "anomaly_thread_spike":"darkorange", "anomaly_combined":"purple",
            "anomaly_io_burst":"green"}
    fig, ax = plt.subplots(figsize=(14, 5))
    offset = 0; xticks, xlbls = [], []
    for rid, data in test_data.items():
        errs, _ = score_sequences(model, data["sequences"], device)
        x = np.arange(offset, offset + len(errs))
        ax.bar(x, errs, color=cmap.get(rid,"gray"), alpha=0.75, width=1.0)
        xticks.append(offset + len(errs)//2)
        xlbls.append(rid.replace("_","\n"))
        offset += len(errs) + 4
    ax.axhline(threshold, color="black", ls="--", lw=1.5, label=f"Threshold={threshold:.4f}")
    ax.set_xticks(xticks); ax.set_xticklabels(xlbls, fontsize=8)
    ax.set_ylabel("Reconstruction Error"); ax.legend()
    ax.set_title("TA-LSTM-AE: Per-Sequence Scores by Run", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/08_dl_anomaly_scores.png", dpi=150); plt.close()

    # ── Plot 3: Confusion matrix ──────────────────────────────────────────
    cm = confusion_matrix(y, yh)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal","Anomaly"], yticklabels=["Normal","Anomaly"])
    ax.set_title("TA-LSTM-AE Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/09_dl_confusion_matrix.png", dpi=150); plt.close()

    # ── Plot 4: ROC curve ─────────────────────────────────────────────────
    if len(set(y)) > 1:
        fpr, tpr, _ = roc_curve(y, ys)
        auc = roc_auc_score(y, ys)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
        ax.plot([0,1],[0,1], "k--", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curve — TA-LSTM-AE", fontweight="bold"); ax.legend()
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/10_dl_roc_curve.png", dpi=150); plt.close()

    # ── Plot 5: Attention heatmap ─────────────────────────────────────────
    picks = [r for r in ["normal_io_01","anomaly_mem_spike"] if r in attn_store]
    if picks:
        fig, axes = plt.subplots(1, len(picks), figsize=(13, 4))
        axes = [axes] if len(picks)==1 else list(axes)
        for ax, rid in zip(axes, picks):
            mat = attn_store[rid][0][:30]
            sns.heatmap(mat, ax=ax, cmap="YlOrRd",
                        xticklabels=range(hp["seq_len"]), yticklabels=False)
            ax.set_title(f"{rid}\n[{'ANOMALY' if attn_store[rid][1] else 'Normal'}]",
                         fontweight="bold")
            ax.set_xlabel("Timestep")
        plt.suptitle("Encoder Attention Weights per Sequence", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/10_dl_attention_heatmap.png", dpi=150); plt.close()

    print(f"\nAll plots -> {FIG_DIR}")


if __name__ == "__main__":
    main()
