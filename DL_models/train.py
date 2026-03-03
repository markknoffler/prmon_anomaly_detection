"""
Training loop for TA-LSTM-AE.
Features: tqdm bars, best-model checkpointing, early stopping, LR scheduler.
"""
import os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from model   import TA_LSTM_AE
from loss    import sequence_normalised_mse
from dataset import build_loaders, FEATURE_COLS

DEFAULTS = dict(
    csv="../data/analysis/combined_dataset.csv",
    out_dir="../data/analysis/dl_results",
    seq_len=10, batch_size=32,
    hidden_dim=64, n_layers=2, dropout=0.2,
    epochs=100, lr=1e-3, patience=15, seed=42,
)


def parse_args():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    return p.parse_args()


def run_epoch(model, loader, optimizer, device, training):
    model.train() if training else model.eval()
    total = 0.0
    ctx   = torch.enable_grad() if training else torch.no_grad()
    bar   = tqdm(loader, desc="  train" if training else "  val", leave=False, ncols=88)
    with ctx:
        for batch in bar:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = sequence_normalised_mse(batch, recon)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
            bar.set_postfix(loss=f"{loss.item():.5f}")
    return total / len(loader)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(os.path.join(args.out_dir, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    tr_loader, va_loader, _, _ = build_loaders(
        args.csv, args.seq_len, args.batch_size,
        scaler_save_path=os.path.join(args.out_dir, "scaler.pkl"),
    )

    model = TA_LSTM_AE(len(FEATURE_COLS), args.hidden_dim, args.n_layers, args.dropout).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5, verbose=True)

    best_val, patience_ctr, history = float("inf"), 0, []
    ckpt_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs", ncols=88):
        tr_loss = run_epoch(model, tr_loader, optimizer, device, training=True)
        va_loss = run_epoch(model, va_loader, optimizer, device, training=False)
        scheduler.step(va_loss)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
        tqdm.write(f"Epoch {epoch:3d} | train={tr_loss:.5f}  val={va_loss:.5f}")

        if va_loss < best_val:
            best_val, patience_ctr = va_loss, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": va_loss, "hparams": vars(args)}, ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                tqdm.write(f"Early stop at epoch {epoch}"); break

    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "training_history.csv"), index=False)
    print(f"Done. Best val loss: {best_val:.6f}  |  Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
