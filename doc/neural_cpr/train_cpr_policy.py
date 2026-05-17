#!/usr/bin/env python3
from __future__ import annotations
"""
Train and export the NeuralCprPolicy MLP.

Architecture:  Dense(14→H, act) → Dense(H→24, linear)
               H controlled by --hidden (default 32)
Activations:   relu | tanh | softplus  (--activation, default relu)
Output:        24 logits over the joint 3×2×2×2 action space
               index i → dec=i//8, cprw=(i//4)%2, fine=(i//2)%2, coarse=i%2
Optimizer:     AdamW with cosine learning-rate decay and early stopping.
Loss:          Label-smoothed cross-entropy with optional inverse-frequency
               class weights (--class-weights) to counter severe imbalance.

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
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# opm-common Kerasify export (optional)
# ---------------------------------------------------------------------------
try:
    from opm.ml.ml_tools.kerasify import export_model
    from opm.ml.ml_tools.dense_layers import Sequential, Dense
    HAS_OPM_ML = True
except ImportError:
    HAS_OPM_ML = False

# ---------------------------------------------------------------------------
# Activation helpers  (must map to activation codes the C++ reader accepts)
# ---------------------------------------------------------------------------

# Binary format activation codes (opm-common ml_model.hpp ActivationType enum)
ACT_CODE = {"linear": 1, "relu": 2, "softplus": 3, "tanh": 6}


def act_forward(name: str, x: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(0.0, x)
    if name == "tanh":
        return np.tanh(x)
    if name == "softplus":
        # numerically stable: log(1+exp(x))
        return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))
    raise ValueError(name)


def act_backward(name: str, pre: np.ndarray, act: np.ndarray) -> np.ndarray:
    """Element-wise derivative of activation w.r.t. its pre-activation input."""
    if name == "relu":
        return (pre > 0).astype(np.float32)
    if name == "tanh":
        return (1.0 - act ** 2).astype(np.float32)
    if name == "softplus":
        # d/dx log(1+exp(x)) = sigmoid(x)
        return (1.0 / (1.0 + np.exp(-np.clip(pre, -20, 20)))).astype(np.float32)
    raise ValueError(name)


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
        # well_density: 0.5% density → 1.0  (0.0 for old CSVs lacking this column)
        c01(float(row.get("well_density", "0.0")) * 200.0),
        # nl_residual_trend: log scale 0.01→0, 1→0.5, 100→1
        float(np.clip(
            (math.log10(max(float(row.get("nl_residual_trend", "1.0")), 0.01)) + 2.0) / 4.0,
            0.0, 1.0)),
        # medium-impact features (0.0 for older CSVs)
        c01(float(row.get("num_phases", "3")) / 3.0),
        c01(float(row.get("dt_cut_count", "0")) / 5.0),
        c01(float(row.get("prev2_linsolver_iters", "0")) / 50.0),
        float(np.clip(
            math.log10(max(float(row.get("condition_number_estimate", "1.0")), 1.0)) / 10.0,
            0.0, 1.0)),
        c01(float(row.get("bhp_well_fraction", "0.0")))
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# MLP  n_features → H×depth → 24  (arbitrary depth)
# ---------------------------------------------------------------------------

class MLP:
    def __init__(self, hidden: int = 32, activation: str = "relu", seed: int = 42,
                 n_features: int = 19, depth: int = 1):
        rng = np.random.default_rng(seed)
        self.activation = activation
        self.depth = depth
        dims = [n_features] + [hidden] * depth + [24]
        self.weights: list[np.ndarray] = []
        self.biases:  list[np.ndarray] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            self.weights.append(
                rng.standard_normal((out_d, in_d)).astype(np.float32) * math.sqrt(2 / in_d))
            self.biases.append(np.zeros(out_d, dtype=np.float32))
        self._init_adam()

    def _init_adam(self):
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t = 0

    def _params(self) -> dict:
        p = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            p[f"W{i}"] = W
            p[f"b{i}"] = b
        return p

    def forward(self, x: np.ndarray) -> tuple:
        pres: list[np.ndarray] = []
        acts: list[np.ndarray] = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            pre = W @ h + b
            pres.append(pre)
            if i < len(self.weights) - 1:
                h = act_forward(self.activation, pre)
            else:
                h = pre  # output layer stays linear
            acts.append(h)
        return acts[-1], (pres, acts)

    def predict(self, x: np.ndarray) -> int:
        logits, _ = self.forward(x)
        return int(np.argmax(logits[:24]))

    def backward(self, cache, y: int,
                 label_smooth: float = 0.1,
                 sample_weight: float = 1.0) -> tuple[float, dict]:
        pres, acts = cache
        logits = acts[-1]

        logits_s = logits - logits.max()
        exp_l    = np.exp(logits_s)
        prob     = exp_l / exp_l.sum()
        n        = len(logits)
        target   = np.full(n, label_smooth / n, dtype=np.float32)
        target[y] += 1.0 - label_smooth
        loss = sample_weight * -(target * np.log(prob + 1e-12)).sum()

        # Backprop through layers in reverse; d_next is grad wrt layer output
        d_next = sample_weight * (prob - target)
        grads: dict[str, np.ndarray] = {}
        for i in range(len(self.weights) - 1, -1, -1):
            pre   = pres[i]
            h_out = acts[i + 1]
            h_in  = acts[i]
            if i < len(self.weights) - 1:
                d_pre = d_next * act_backward(self.activation, pre, h_out)
            else:
                d_pre = d_next  # output layer is linear
            grads[f"W{i}"] = np.outer(d_pre, h_in)
            grads[f"b{i}"] = d_pre
            d_next = self.weights[i].T @ d_pre
        return loss, grads

    def adam_step(self, grads: dict, lr: float,
                  beta1: float = 0.9, beta2: float = 0.999,
                  eps: float = 1e-8, wd: float = 1e-4):
        self._t += 1
        t = self._t
        for k, g in grads.items():
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * g * g
            m_hat = self._m[k] / (1 - beta1 ** t)
            v_hat = self._v[k] / (1 - beta2 ** t)
            p = self._params()[k]
            decay = wd if k.startswith("W") else 0.0
            p -= lr * (m_hat / (np.sqrt(v_hat) + eps) + decay * p)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def make_class_weights(labels: list[int], n_classes: int = 24,
                       cap: float = 10.0) -> np.ndarray:
    """Inverse-frequency weights, capped at cap × mean weight."""
    counts = Counter(labels)
    n = len(labels)
    raw = np.array([n / max(counts.get(c, 1), 1) for c in range(n_classes)],
                   dtype=np.float64)
    mean_w = raw.mean()
    raw = np.minimum(raw, cap * mean_w)
    # Normalise so mean = 1
    return (raw / raw.mean()).astype(np.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def cosine_lr(epoch: int, total: int, lr_max: float, lr_min: float = 1e-5) -> float:
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / total))


def train(features: list[np.ndarray], labels: list[int],
          val_features: list[np.ndarray] | None = None,
          val_labels:   list[int]        | None = None,
          epochs: int        = 300,
          lr_max: float      = 3e-3,
          patience: int      = 30,
          seed: int          = 42,
          hidden: int        = 32,
          depth: int         = 1,
          activation: str    = "relu",
          class_weights: np.ndarray | None = None,
          noise_std: float   = 0.0,
          n_features: int    = 19) -> MLP:

    model = MLP(hidden=hidden, depth=depth, activation=activation, seed=seed, n_features=n_features)
    rng   = np.random.default_rng(seed)          # ← NEW: rng for noise
    n = len(features)
    idx = np.arange(n)
    best_val_loss = float("inf")
    best_state: dict | None = None
    no_improve = 0

    for epoch in range(epochs):
        lr = cosine_lr(epoch, epochs, lr_max)
        rng.shuffle(idx)
        total_loss = 0.0

        for i in idx:
            sw = float(class_weights[labels[i]]) if class_weights is not None else 1.0
            # ↓ NEW: add Gaussian noise to features, clip to keep in [0,1]
            x = features[i]
            if noise_std > 0.0:
                x = np.clip(x + rng.normal(0.0, noise_std, size=x.shape).astype(np.float32),
                            0.0, 1.0)
            _, cache = model.forward(x)
            loss, grads = model.backward(cache, labels[i], sample_weight=sw)
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
        params = model._params()
        for k, v in best_state.items():
            params[k][...] = v
        print(f"  Restored best checkpoint (val_loss={best_val_loss:.4f})")

    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_opm_kerasify(model: MLP, out_path: str):
    if not HAS_OPM_ML:
        raise ImportError(
            "opm.ml not found — use --format=binary instead."
        )
    n_layers = len(model.weights)
    dims = [model.weights[0].shape[1]] + [W.shape[0] for W in model.weights]
    acts = [model.activation] * (n_layers - 1) + ["linear"]
    layers = [Dense(dims[i], dims[i+1], acts[i]) for i in range(n_layers)]
    for i, layer in enumerate(layers):
        layer.weights = model.weights[i].T
        layer.biases  = model.biases[i]
    export_model(Sequential(layers), out_path)
    print(f"Exported Kerasify model → {out_path}")


def export_binary_fallback(model: MLP, out_path: str):
    LAYER_DENSE = 3
    ACT_LINEAR  = ACT_CODE["linear"]
    act_code    = ACT_CODE[model.activation]
    n_layers    = len(model.weights)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", n_layers))
        for i, (W, b) in enumerate(zip(model.weights, model.biases)):
            act = ACT_LINEAR if i == n_layers - 1 else act_code
            out_dim, in_dim = W.shape
            f.write(struct.pack("<I", LAYER_DENSE))
            f.write(struct.pack("<I", in_dim))
            f.write(struct.pack("<I", out_dim))
            f.write(struct.pack("<I", out_dim))
            f.write(W.T.astype(np.float32).tobytes())
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
    parser.add_argument("--data",         required=True)
    parser.add_argument("--out",          default="cpr_policy.model")
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--lr",           type=float, default=3e-3)
    parser.add_argument("--val-frac",     type=float, default=0.15)
    parser.add_argument("--patience",     type=int,   default=40)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--format",       choices=["kerasify", "binary"],
                        default="binary")
    parser.add_argument("--hidden",       type=int,   default=32,
                        help="Hidden layer width (default 32)")
    parser.add_argument("--depth",        type=int,   default=1,
                        help="Number of hidden layers (default 1)")
    parser.add_argument("--activation",   choices=list(ACT_CODE.keys()),
                        default="relu",
                        help="Hidden layer activation (default relu)")
    parser.add_argument("--class-weights", action="store_true",
                        help="Weight loss by inverse class frequency")
    parser.add_argument("--weight-cap",   type=float, default=10.0,
                        help="Max class weight relative to mean (default 10)")
    parser.add_argument("--noise-std",    type=float, default=0.0,   # ← NEW
                        help="Std of Gaussian noise added to features during training "
                             "(0 = off, 0.02 recommended)")
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

    cw = None
    if args.class_weights:
        cw = make_class_weights(tr_l, cap=args.weight_cap)
        print(f"  Class weights: min={cw[cw>0].min():.2f}  "
              f"max={cw.max():.2f}  effective={int((cw > 0).sum())}/24  "
              f"(cap={args.weight_cap}×mean)")

    n_features = len(tr_f[0]) if tr_f else 19
    hidden_str = "→".join([str(args.hidden)] * args.depth)
    arch = f"{n_features}→{hidden_str}→24"
    print(f"Training ({arch}, {args.activation}, AdamW, patience={args.patience}"
          + (", class-weighted" if cw is not None else "")
          + (f", noise_std={args.noise_std}" if args.noise_std > 0 else "")
          + ") ...")
    model = train(tr_f, tr_l, va_f, va_l,
                  epochs=args.epochs, lr_max=args.lr,
                  patience=args.patience, seed=args.seed,
                  hidden=args.hidden, depth=args.depth,
                  activation=args.activation,
                  class_weights=cw,
                  noise_std=args.noise_std,
                  n_features=n_features)

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
