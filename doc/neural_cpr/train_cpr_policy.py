#!/usr/bin/env python3
"""
Train and export the NeuralCprPolicy MLP.

Architecture:  Dense(14→32, relu) → Dense(32→24, linear)
Output:        24 logits over the joint 3×2×2×2 action space
               index i → dec=i//8, cprw=(i//4)%2, fine=(i//2)%2, coarse=i%2
Optimizer:     AdamW with cosine learning-rate decay and early stopping.

Input CSV
---------
Produced by collect_sweep_labels.py.  Required columns:
  dt_days, dt_ratio, nl_residual_norm, nl_residual_reduce, nl_iteration,
  nnz_per_row, diag_dominance, prev_linsolver_iters, prev_solve_failed,
  time_elapsed_frac, num_cells_log, block_size, label (int 0-23)

Usage
-----
python train_cpr_policy.py --data training_data.csv --out cpr_policy.model

The exported file is loaded at runtime via:
    export OPM_NEURAL_CPR_WEIGHTS=/path/to/cpr_policy.model
"""

import argparse
import csv
import math
import os
import struct

import numpy as np

# ---------------------------------------------------------------------------
# opm-common Kerasify export (optional — only needed for model export)
# ---------------------------------------------------------------------------
try:
    from opm.ml.ml_tools.kerasify import export_model
    from opm.ml.ml_tools.dense_layers import Sequential, Dense
    HAS_OPM_ML = True
except ImportError:
    HAS_OPM_ML = False

# ---------------------------------------------------------------------------
# Feature normalisation  (must stay in sync with CprPolicyFeatures::toArray())
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "dt_days", "dt_ratio", "nl_residual_norm", "nl_residual_reduce",
    "nl_iteration", "nnz_per_row", "diag_dominance",
    "prev_linsolver_iters", "prev_solve_failed",
    "time_elapsed_frac", "num_cells_log", "block_size",
]


def normalise(row: dict) -> np.ndarray:
    def c01(v):
        return float(np.clip(v, 0.0, 1.0))

    return np.array([
        math.log1p(float(row["dt_days"])) / 7.0,
        c01(float(row["dt_ratio"]) / 10.0),
        (math.log10(max(float(row["nl_residual_norm"]), 1e-14)) + 14.0) / 14.0,
        c01(float(row["nl_residual_reduce"])),
        c01(float(row["nl_iteration"]) / 20.0),
        math.log1p(float(row["nnz_per_row"])) / 5.0,
        c01(float(row["diag_dominance"]) / 4.0),
        c01(float(row["prev_linsolver_iters"]) / 50.0),
        float(np.clip(float(row["prev_solve_failed"]), 0.0, 1.0)),
        c01(float(row["time_elapsed_frac"])),
        c01(float(row["num_cells_log"]) / 6.0),
        c01(float(row["block_size"]) / 6.0),
        0.0,   # pad0
        0.0,   # pad1
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# MLP  14→32→24
# ---------------------------------------------------------------------------

class MLP:
    """Two-layer MLP: 14→32 (relu) → 32→24 (linear).

    Smaller than the previous 3-layer design; better regularised for the
    ~5 k tabular training samples typical of a CPR sweep dataset.
    Parameter count: 14×32+32 + 32×24+24 = 1 272.
    """

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((32, 14)).astype(np.float32) * math.sqrt(2 / 14)
        self.b1 = np.zeros(32, dtype=np.float32)
        self.W2 = rng.standard_normal((24, 32)).astype(np.float32) * math.sqrt(2 / 32)
        self.b2 = np.zeros(24, dtype=np.float32)
        self._init_adam()

    def _init_adam(self):
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t = 0

    def _params(self):
        return {"W1": self.W1, "b1": self.b1,
                "W2": self.W2, "b2": self.b2}

    def forward(self, x: np.ndarray) -> tuple:
        h_pre = self.W1 @ x + self.b1
        h     = np.maximum(0, h_pre)
        logits = self.W2 @ h + self.b2
        return logits, (x, h_pre, h)

    def predict(self, x: np.ndarray) -> int:
        logits, _ = self.forward(x)
        return int(np.argmax(logits[:24]))

    def backward(self, cache, y: int, label_smooth: float = 0.1) -> tuple[float, dict]:
        x, h_pre, h = cache
        logits = self.W2 @ h + self.b2

        logits_s = logits - logits.max()
        exp  = np.exp(logits_s)
        prob = exp / exp.sum()
        n    = len(logits)
        target = np.full(n, label_smooth / n, dtype=np.float32)
        target[y] += 1.0 - label_smooth
        loss = -(target * np.log(prob + 1e-12)).sum()

        d_logits = prob - target
        dW2 = np.outer(d_logits, h)
        db2 = d_logits
        dh  = self.W2.T @ d_logits
        dh_pre = dh * (h_pre > 0)
        dW1 = np.outer(dh_pre, x)
        db1 = dh_pre

        return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def adam_step(self, grads: dict, lr: float,
                  beta1: float = 0.9, beta2: float = 0.999,
                  eps: float = 1e-8, wd: float = 1e-4):
        self._t += 1
        t = self._t
        params = self._params()
        for k, g in grads.items():
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * g * g
            m_hat = self._m[k] / (1 - beta1 ** t)
            v_hat = self._v[k] / (1 - beta2 ** t)
            decay = wd if k.startswith("W") else 0.0
            params[k] -= lr * (m_hat / (np.sqrt(v_hat) + eps) + decay * params[k])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def cosine_lr(epoch: int, total: int, lr_max: float, lr_min: float = 1e-5) -> float:
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / total))


def train(features: list[np.ndarray], labels: list[int],
          val_features: list[np.ndarray] | None = None,
          val_labels:   list[int]        | None = None,
          epochs: int   = 300,
          lr_max: float = 3e-3,
          patience: int = 30,
          seed: int     = 42) -> MLP:

    model = MLP(seed=seed)
    n = len(features)
    idx = np.arange(n)
    best_val_loss = float("inf")
    best_state: dict | None = None
    no_improve = 0

    for epoch in range(epochs):
        lr = cosine_lr(epoch, epochs, lr_max)
        np.random.shuffle(idx)
        total_loss = 0.0

        for i in idx:
            _, cache = model.forward(features[i])
            loss, grads = model.backward(cache, labels[i])
            model.adam_step(grads, lr)
            total_loss += loss

        if val_features is not None:
            vl = sum(
                model.backward(model.forward(f)[1], l)[0]
                for f, l in zip(val_features, val_labels)
            ) / len(val_features)

            if vl < best_val_loss - 1e-4:
                best_val_loss = vl
                best_state = {k: v.copy() for k, v in model._params().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 20 == 0:
                tr_acc = sum(model.predict(f) == l
                             for f, l in zip(features, labels)) / n
                va_acc = sum(model.predict(f) == l
                             for f, l in zip(val_features, val_labels)
                             ) / len(val_features)
                print(f"  epoch {epoch+1:4d}/{epochs}  lr={lr:.2e}  "
                      f"loss={total_loss/n:.4f}  val_loss={vl:.4f}  "
                      f"tr_acc={tr_acc:.3f}  val_acc={va_acc:.3f}")

            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1} "
                      f"(no val improvement for {patience} epochs)")
                break
        else:
            if (epoch + 1) % 20 == 0:
                tr_acc = sum(model.predict(f) == l
                             for f, l in zip(features, labels)) / n
                print(f"  epoch {epoch+1:4d}/{epochs}  lr={lr:.2e}  "
                      f"loss={total_loss/n:.4f}  acc={tr_acc:.3f}")

    if best_state is not None:
        for k, v in best_state.items():
            getattr(model, k)[...] = v
        print(f"  Restored best checkpoint (val_loss={best_val_loss:.4f})")

    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_opm_kerasify(model: MLP, out_path: str):
    if not HAS_OPM_ML:
        raise ImportError(
            "opm.ml not found — use --format=binary instead.\n"
            "Install opm-common's Python package to enable Kerasify export."
        )
    layers = [Dense(14, 32, "relu"), Dense(32, 24, "linear")]
    layers[0].weights = model.W1.T   # Dense stores (in, out)
    layers[0].biases  = model.b1
    layers[1].weights = model.W2.T
    layers[1].biases  = model.b2
    export_model(Sequential(layers), out_path)
    print(f"Exported Kerasify model → {out_path}")


def export_binary_fallback(model: MLP, out_path: str):
    """Write raw Kerasify binary without opm.ml dependency."""
    LAYER_DENSE = 3
    ACT_LINEAR  = 1
    ACT_RELU    = 2

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", 2))  # num_layers
        for (W, b), act in zip(
            [(model.W1, model.b1), (model.W2, model.b2)],
            [ACT_RELU, ACT_LINEAR],
        ):
            out_dim, in_dim = W.shape
            f.write(struct.pack("<I", LAYER_DENSE))
            f.write(struct.pack("<I", in_dim))
            f.write(struct.pack("<I", out_dim))
            f.write(struct.pack("<I", out_dim))
            f.write(W.T.astype(np.float32).tobytes())  # (in, out) row-major
            f.write(b.astype(np.float32).tobytes())
            f.write(struct.pack("<I", act))

    print(f"Exported binary model → {out_path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> tuple[list[np.ndarray], list[int]]:
    features, labels = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            features.append(normalise(row))
            labels.append(int(row["label"]))
    return features, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data",     required=True,
                        help="CSV from collect_sweep_labels.py")
    parser.add_argument("--out",      default="cpr_policy.model")
    parser.add_argument("--epochs",   type=int,   default=300)
    parser.add_argument("--lr",       type=float, default=3e-3,
                        help="Peak learning rate for cosine schedule")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="Validation fraction for early stopping")
    parser.add_argument("--patience", type=int,   default=30)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--format",   choices=["kerasify", "binary"],
                        default="kerasify")
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    features, labels = load_csv(args.data)
    n = len(features)
    print(f"  {n} samples, {len(set(labels))} distinct labels (of 24)")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    val_n = max(1, int(n * args.val_frac))
    val_idx, tr_idx = perm[:val_n], perm[val_n:]
    tr_f = [features[i] for i in tr_idx]
    tr_l = [labels[i]   for i in tr_idx]
    va_f = [features[i] for i in val_idx]
    va_l = [labels[i]   for i in val_idx]
    print(f"  Train: {len(tr_f)}  Val: {len(va_f)}")

    print(f"Training (14→32→24, AdamW, cosine LR, patience={args.patience}) ...")
    model = train(tr_f, tr_l, va_f, va_l,
                  epochs=args.epochs, lr_max=args.lr,
                  patience=args.patience, seed=args.seed)

    if args.format == "kerasify":
        export_opm_kerasify(model, args.out)
    else:
        export_binary_fallback(model, args.out)

    tr_acc = sum(model.predict(f) == l for f, l in zip(tr_f, tr_l)) / len(tr_f)
    va_acc = sum(model.predict(f) == l for f, l in zip(va_f, va_l)) / len(va_f)
    print(f"Final  train acc: {tr_acc:.3f}  val acc: {va_acc:.3f}")
    print(f"\nTo use:\n  export OPM_NEURAL_CPR_WEIGHTS={os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
