#!/usr/bin/env python3
from __future__ import annotations
"""
Phase-1 pre-training for the NeuralCprPolicy encoder.

Trains a shared feature encoder on an auxiliary regression task:
    features (19) + label one-hot (24)  →  log(total_ms)

This uses all raw sweep CSV rows (~1M) rather than the ~12k labeled
samples from collect_sweep_labels.py, giving the encoder a richer signal
about how each solver configuration performs in different simulation states.

The learned encoder weights are saved to a .npz file and used by
train_cpr_policy.py via --pretrained to warm-start the classifier instead
of random initialisation.

Architecture
------------
  encoder:   n_features → H  (depth hidden layers, ReLU)   ← transferred
  reg_head:  (H + 24)  → 1   (linear, predicts log total_ms) ← discarded

Usage
-----
python pretrain_cpr_policy.py \\
    --csvs cpr_sweep_data/sweep2/*.csv cpr_sweep_data/sweep_spe11/*.csv \\
    --out  tmp/cpr_encoder_pretrained.npz \\
    --hidden 24 --depth 1 --epochs 30 --seed 42
"""

import argparse
import csv as csv_mod
import math
import os
import re
import sys

import numpy as np

# Reuse normalisation and LR schedule from train_cpr_policy (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cpr_policy import normalise, cosine_lr

N_LABELS = 24
BATCH    = 512


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _label_from_path(path: str) -> int | None:
    """Extract label 0-23 from filename suffix like DECK_07.csv → 7."""
    m = re.search(r'_(\d{2})\.csv$', os.path.basename(path))
    return int(m.group(1)) if m else None


def load_data(csv_paths: list[str],
              max_rows: int | None = None,
              rng: np.random.Generator | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X        : float32 (N, n_features)  — normalised features
    L        : float32 (N, N_LABELS)    — one-hot label encoding
    log_ms   : float32 (N,)             — log(total_ms) regression target
    """
    X_list, L_list, T_list = [], [], []
    skipped = 0
    for path in csv_paths:
        lbl = _label_from_path(path)
        if lbl is None or not (0 <= lbl < N_LABELS):
            skipped += 1
            continue
        oh = np.zeros(N_LABELS, dtype=np.float32)
        oh[lbl] = 1.0
        with open(path, newline='') as fh:
            for row in csv_mod.DictReader(fh):
                try:
                    ms = float(row['total_ms'])
                    if ms <= 0:
                        continue
                    X_list.append(normalise(row))
                    L_list.append(oh)
                    T_list.append(math.log(ms))
                except (KeyError, ValueError):
                    continue

    if skipped:
        print(f"  Warning: {skipped} files skipped (no label in filename)")

    X = np.array(X_list, dtype=np.float32)
    L = np.array(L_list, dtype=np.float32)
    T = np.array(T_list, dtype=np.float32)

    if max_rows is not None and len(X) > max_rows:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(len(X), max_rows, replace=False)
        X, L, T = X[idx], L[idx], T[idx]

    return X, L, T


# ---------------------------------------------------------------------------
# Pre-training MLP  (vectorised mini-batch)
# ---------------------------------------------------------------------------

class PretrainMLP:
    """
    encoder:   n_features → H  (depth ReLU layers)   — weights transferred
    reg_head:  (H + 24)   → 1  (linear)               — discarded after training
    """

    def __init__(self, hidden: int, depth: int, n_features: int, seed: int):
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        self.depth  = depth

        # Encoder layers
        dims = [n_features] + [hidden] * depth
        self.enc_W = [
            (rng.standard_normal((dims[i+1], dims[i])) * math.sqrt(2 / dims[i])).astype(np.float32)
            for i in range(len(dims) - 1)
        ]
        self.enc_b = [np.zeros(hidden, dtype=np.float32) for _ in range(depth)]

        # Regression head: concat(encoded, label_onehot) → 1
        reg_in = hidden + N_LABELS
        self.reg_W = (rng.standard_normal((1, reg_in)) * math.sqrt(2 / reg_in)).astype(np.float32)
        self.reg_b = np.zeros(1, dtype=np.float32)

        self._init_adam()

    # ── Adam state ────────────────────────────────────────────────────────

    def _all_params(self) -> dict[str, np.ndarray]:
        p: dict[str, np.ndarray] = {}
        for i, (W, b) in enumerate(zip(self.enc_W, self.enc_b)):
            p[f"eW{i}"] = W
            p[f"eb{i}"] = b
        p["rW"] = self.reg_W
        p["rb"] = self.reg_b
        return p

    def _init_adam(self):
        self._m = {k: np.zeros_like(v) for k, v in self._all_params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._all_params().items()}
        self._t = 0

    # ── Forward  (batch) ─────────────────────────────────────────────────

    def forward(self, X: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, list]:
        """
        X : (B, n_features)
        L : (B, N_LABELS)
        returns pred (B,) and cache for backward
        """
        pres, acts = [], [X]
        H = X
        for W, b in zip(self.enc_W, self.enc_b):
            pre = H @ W.T + b          # (B, hidden)
            pres.append(pre)
            H = np.maximum(0.0, pre)   # ReLU
            acts.append(H)

        combined = np.concatenate([H, L], axis=1)          # (B, hidden+24)
        pred = (combined @ self.reg_W.T + self.reg_b)[:, 0]  # (B,)
        return pred, (pres, acts, combined)

    # ── Backward (batch, MSE) ────────────────────────────────────────────

    def backward(self, cache, target: np.ndarray) -> tuple[float, dict]:
        """target : (B,)"""
        pres, acts, combined = cache
        B = len(target)

        pred = (combined @ self.reg_W.T + self.reg_b)[:, 0]
        residual = pred - target                           # (B,)
        loss = 0.5 * float((residual ** 2).mean())

        d_pred = (residual / B)[:, np.newaxis]             # (B, 1)
        d_reg_W = d_pred.T @ combined                      # (1, hidden+24)
        d_reg_b = d_pred.sum(axis=0)                       # (1,)
        d_combined = d_pred @ self.reg_W                   # (B, hidden+24)

        d_H = d_combined[:, :self.hidden]                  # (B, hidden)

        grads: dict[str, np.ndarray] = {"rW": d_reg_W, "rb": d_reg_b}
        for i in range(self.depth - 1, -1, -1):
            pre   = pres[i]                                # (B, hidden)
            H_in  = acts[i]                                # (B, in_dim)
            d_pre = d_H * (pre > 0).astype(np.float32)    # ReLU backward
            grads[f"eW{i}"] = d_pre.T @ H_in              # (hidden, in_dim)
            grads[f"eb{i}"] = d_pre.sum(axis=0)           # (hidden,)
            d_H = d_pre @ self.enc_W[i]                    # (B, in_dim)

        return loss, grads

    # ── Adam step ────────────────────────────────────────────────────────

    def adam_step(self, grads: dict, lr: float,
                  beta1: float = 0.9, beta2: float = 0.999,
                  eps: float = 1e-8, wd: float = 1e-4):
        self._t += 1
        t = self._t
        params = self._all_params()
        for k, g in grads.items():
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * g * g
            m_hat = self._m[k] / (1 - beta1 ** t)
            v_hat = self._v[k] / (1 - beta2 ** t)
            p = params[k]
            decay = wd if k.startswith(("eW", "rW")) else 0.0
            p -= lr * (m_hat / (np.sqrt(v_hat) + eps) + decay * p)

    # ── Save encoder ─────────────────────────────────────────────────────

    def save_encoder(self, path: str):
        """Save only encoder weights (discards regression head)."""
        npz = {f"W{i}": W for i, W in enumerate(self.enc_W)}
        npz.update({f"b{i}": b for i, b in enumerate(self.enc_b)})
        np.savez(path, **npz)
        n_params = sum(W.size + b.size for W, b in zip(self.enc_W, self.enc_b))
        print(f"Saved encoder → {path}  "
              f"({self.depth} layers, hidden={self.hidden}, {n_params} params)")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(X: np.ndarray, L: np.ndarray, T_raw: np.ndarray,
          hidden: int, depth: int, epochs: int,
          lr_max: float, seed: int, patience: int = 15) -> PretrainMLP:

    rng = np.random.default_rng(seed)
    n   = len(X)

    # Normalise targets to zero-mean unit-variance for stable training
    t_mean = T_raw.mean()
    t_std  = T_raw.std() + 1e-8
    T = (T_raw - t_mean) / t_std

    # Train / val split (10% val)
    perm  = rng.permutation(n)
    val_n = max(1, int(n * 0.10))
    val_idx = perm[:val_n]
    tr_idx  = perm[val_n:]
    X_tr, L_tr, T_tr = X[tr_idx], L[tr_idx], T[tr_idx]
    X_va, L_va, T_va = X[val_idx], L[val_idx], T[val_idx]

    n_features = X.shape[1]
    model = PretrainMLP(hidden=hidden, depth=depth, n_features=n_features, seed=seed)

    best_val  = float("inf")
    best_enc  = None
    no_improve = 0
    n_tr = len(X_tr)

    for epoch in range(epochs):
        lr  = cosine_lr(epoch, epochs, lr_max)
        idx = rng.permutation(n_tr)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, n_tr, BATCH):
            bi = idx[start:start + BATCH]
            pred, cache = model.forward(X_tr[bi], L_tr[bi])
            loss, grads = model.backward(cache, T_tr[bi])
            model.adam_step(grads, lr)
            total_loss += loss
            n_batches  += 1

        # Validation MAE (in normalised target space)
        val_pred, _ = model.forward(X_va, L_va)
        val_mae = float(np.abs(val_pred - T_va).mean())

        if val_mae < best_val - 1e-5:
            best_val   = val_mae
            best_enc   = [(W.copy(), b.copy())
                          for W, b in zip(model.enc_W, model.enc_b)]
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  lr={lr:.2e}  "
                  f"loss={total_loss/n_batches:.4f}  val_mae={val_mae:.4f}"
                  + (" *" if no_improve == 0 else ""))

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch + 1}")
            break

    if best_enc is not None:
        for i, (W, b) in enumerate(best_enc):
            model.enc_W[i][...] = W
            model.enc_b[i][...] = b
        print(f"  Restored best encoder (val_mae={best_val:.4f})")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csvs",     nargs="+", required=True,
                        help="Raw sweep CSV files (may include globs)")
    parser.add_argument("--out",      required=True,
                        help="Output .npz file for encoder weights")
    parser.add_argument("--hidden",   type=int,   default=24)
    parser.add_argument("--depth",    type=int,   default=1)
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--lr",       type=float, default=3e-3)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--max-rows", type=int,   default=None,
                        help="Random subsample of rows (default: all)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Loading {len(args.csvs)} sweep CSVs ...")
    X, L, T = load_data(args.csvs, max_rows=args.max_rows, rng=rng)
    n_features = X.shape[1]
    print(f"  {len(X):,} rows  ({n_features} features, {N_LABELS} labels)")
    if len(X) == 0:
        print("ERROR: no data loaded"); return

    hidden_str = "→".join([str(args.hidden)] * args.depth)
    print(f"Pre-training encoder ({n_features}→{hidden_str}, "
          f"epochs={args.epochs}, batch={BATCH}) ...")
    model = train(X, L, T,
                  hidden=args.hidden, depth=args.depth,
                  epochs=args.epochs, lr_max=args.lr, seed=args.seed)
    model.save_encoder(args.out)
    print(f"\nTo use:  train_cpr_policy.py --pretrained {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
