"""
train.py
Trains the GentrificationLSTM and saves the best checkpoint.
Run: python train.py
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from lstm_model import GentrificationLSTM, EarlyStopping

# ── Config ──────────────────────────────────────────────────────────────────
PROC_DIR  = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent / "checkpoints"
MODEL_DIR.mkdir(exist_ok=True)

EPOCHS      = 80
BATCH_SIZE  = 32
LR          = 1e-3
VAL_SPLIT   = 0.2
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load data ────────────────────────────────────────────────────────────────
def load_data():
    X = np.load(PROC_DIR / "X.npy")
    y = np.load(PROC_DIR / "y.npy")
    feature_names = (
        open(PROC_DIR / "feature_names.csv").read().strip().split("\n")[1:]
    )
    return X, y, feature_names


# ── Training loop ────────────────────────────────────────────────────────────
def train():
    print(f"Device: {DEVICE}")

    X, y, feature_names = load_data()
    print(f"Data: X={X.shape}, y={y.shape}, pos_rate={y.mean():.2%}")

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset   = TensorDataset(X_t, y_t)
    val_size  = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = GentrificationLSTM(
        input_size=X.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    # Weighted BCE to handle class imbalance
    pos_weight = torch.tensor([(1 - y.mean()) / y.mean()]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    stopper    = EarlyStopping(patience=15)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print("\nTraining...\n")
    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= train_size

        # ── Validate ──
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
                correct  += ((pred > 0.5) == yb).sum().item()
                total    += len(yb)
        val_loss /= val_size
        val_acc   = correct / total

        scheduler.step(val_loss)
        stopper(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")

        if stopper.should_stop:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    # ── Save metadata ──
    meta = {
        "input_size":   X.shape[2],
        "hidden_size":  HIDDEN_SIZE,
        "num_layers":   NUM_LAYERS,
        "dropout":      DROPOUT,
        "feature_names": feature_names,
        "best_val_loss": best_val_loss,
        "final_val_acc": history["val_acc"][-1],
    }
    with open(MODEL_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Best val loss: {best_val_loss:.4f}")
    print(f"✓ Model saved to model/checkpoints/best_model.pt")


if __name__ == "__main__":
    train()
