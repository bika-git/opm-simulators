#!/usr/bin/env python3
"""
Training script for the NeuralCprPolicy MLP.

Architecture:  Dense(14→32, relu) → Dense(32→16, relu) → Dense(16→24, linear)
Output:        24 logits over the joint 3×2×2×2 action space
               index i → dec=i//8, cprw=(i//4)%2, fine=(i//2)%2, coarse=i%2

Exports a Kerasify binary model file consumable by Opm::ML::NNModel<float>
via opm-common's opm.ml.ml_tools.kerasify.export_model().

Usage
-----
1.  Collect sweep data (see "Data collection" below).
2.  pip install numpy  (no TensorFlow needed — opm-common layers are numpy-based)
3.  python train_cpr_policy.py --data sweep_data.csv --out cpr_policy.model

The exported file is loaded at runtime by setting:
    export OPM_NEURAL_CPR_WEIGHTS=/path/to/cpr_policy.model
"""

import argparse
import csv
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Check for opm-common Python package (needed for export only)
# ---------------------------------------------------------------------------
try:
    from opm.ml.ml_tools.kerasify import export_model
    from opm.ml.ml_tools.dense_layers import Sequential, Dense
    HAS_OPM_ML = True
except ImportError:
    HAS_OPM_ML = False

# ---------------------------------------------------------------------------
# Feature normalisation  (must match CprPolicyFeatures::toArray() in C++)
# ---------------------------------------------------------------------------

def normalise(row: dict) -> np.ndarray:
    """Map raw feature values to ~[0,1].  Mirrors CprPolicyFeatures::toArray()."""
    def c01(v):
        return float(np.clip(v, 0.0, 1.0))

    dt_days              = float(row["dt_days"])
    dt_ratio             = float(row["dt_ratio"])
    nl_residual_norm     = float(row["nl_residual_norm"])
    nl_residual_reduce   = float(row["nl_residual_reduce"])
    nl_iteration         = float(row["nl_iteration"])
    nnz_per_row          = float(row["nnz_per_row"])
    diag_dominance       = float(row["diag_dominance"])
    prev_linsolver_iters = float(row["prev_linsolver_iters"])
    prev_solve_failed    = float(row["prev_solve_failed"])
    time_elapsed_frac    = float(row["time_elapsed_frac"])
    num_cells_log        = float(row["num_cells_log"])
    block_size           = float(row["block_size"])

    return np.array([
        math.log1p(dt_days) / 7.0,
        c01(dt_ratio / 10.0),
        (math.log10(max(nl_residual_norm, 1e-14)) + 14.0) / 14.0,
        c01(nl_residual_reduce),
        c01(nl_iteration / 20.0),
        math.log1p(nnz_per_row) / 5.0,
        c01(diag_dominance / 4.0),
        c01(prev_linsolver_iters / 50.0),
        float(np.clip(prev_solve_failed, 0.0, 1.0)),
        c01(time_elapsed_frac),
        c01(num_cells_log / 6.0),
        c01(block_size / 6.0),
        0.0,   # pad0
        0.0,   # pad1
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Action encoding  (must match NeuralCprPolicy::decodeLogits() in C++)
# ---------------------------------------------------------------------------
# Joint index: dec*8 + cprw*4 + fine*2 + coarse
#   dec    ∈ {0=QuasiIMPES, 1=TrueIMPES, 2=TrueIMPESAnalytic}
#   cprw   ∈ {0=cpr, 1=cprw}
#   fine   ∈ {0=ParOverILU0, 1=DILU}
#   coarse ∈ {0=ILU0, 1=DILU}

WEIGHT_TYPE = {"quasiimpes": 0, "trueimpes": 1, "trueimpesanalytic": 2}
CPRW        = {"cpr": 0, "cprw": 1}
FINE        = {"paroverilu0": 0, "dilu": 1}
COARSE      = {"ilu0": 0, "dilu": 1}


def action_index(weight_type: str, use_cprw: str,
                 fine_smoother: str, coarse_smoother: str) -> int:
    dec    = WEIGHT_TYPE[weight_type.lower()]
    cprw   = CPRW[use_cprw.lower()]
    fine   = FINE[fine_smoother.lower()]
    coarse = COARSE[coarse_smoother.lower()]
    return dec * 8 + cprw * 4 + fine * 2 + coarse


# ---------------------------------------------------------------------------
# Numpy MLP  (forward pass, for inference validation)
# ---------------------------------------------------------------------------

class MLP:
    """Tiny 14→32→16→24 MLP with ReLU hidden activations."""

    def __init__(self):
        rng = np.random.default_rng(42)
        # He initialisation
        self.W1 = rng.standard_normal((32, 14)).astype(np.float32) * math.sqrt(2 / 14)
        self.b1 = np.zeros(32, dtype=np.float32)
        self.W2 = rng.standard_normal((16, 32)).astype(np.float32) * math.sqrt(2 / 32)
        self.b2 = np.zeros(16, dtype=np.float32)
        self.W3 = rng.standard_normal((24, 16)).astype(np.float32) * math.sqrt(2 / 16)
        self.b3 = np.zeros(24, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (14,)  →  logits: (24,)"""
        h1 = np.maximum(0, self.W1 @ x + self.b1)
        h2 = np.maximum(0, self.W2 @ h1 + self.b2)
        return self.W3 @ h2 + self.b3

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)))


# ---------------------------------------------------------------------------
# Training loop (SGD with momentum)
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits: np.ndarray, target: int) -> tuple[float, np.ndarray]:
    logits = logits - logits.max()
    exp    = np.exp(logits)
    prob   = exp / exp.sum()
    loss   = -math.log(prob[target] + 1e-12)
    grad   = prob.copy()
    grad[target] -= 1.0
    return loss, grad


def train(features: list[np.ndarray], labels: list[int],
          epochs: int = 200, lr: float = 0.01, momentum: float = 0.9) -> MLP:
    model = MLP()
    n = len(features)

    # Momentum buffers
    vW1 = np.zeros_like(model.W1); vb1 = np.zeros_like(model.b1)
    vW2 = np.zeros_like(model.W2); vb2 = np.zeros_like(model.b2)
    vW3 = np.zeros_like(model.W3); vb3 = np.zeros_like(model.b3)

    idx = np.arange(n)
    for epoch in range(epochs):
        np.random.shuffle(idx)
        total_loss = 0.0
        for i in idx:
            x, y = features[i], labels[i]

            # Forward
            h1_pre = model.W1 @ x + model.b1
            h1 = np.maximum(0, h1_pre)
            h2_pre = model.W2 @ h1 + model.b2
            h2 = np.maximum(0, h2_pre)
            logits = model.W3 @ h2 + model.b3

            loss, d_logits = cross_entropy_loss(logits, y)
            total_loss += loss

            # Backward
            dW3 = np.outer(d_logits, h2)
            db3 = d_logits
            dh2 = model.W3.T @ d_logits
            dh2_pre = dh2 * (h2_pre > 0)
            dW2 = np.outer(dh2_pre, h1)
            db2 = dh2_pre
            dh1 = model.W2.T @ dh2_pre
            dh1_pre = dh1 * (h1_pre > 0)
            dW1 = np.outer(dh1_pre, x)
            db1 = dh1_pre

            # SGD + momentum update
            for param, grad, v in [
                (model.W1, dW1, vW1), (model.b1, db1, vb1),
                (model.W2, dW2, vW2), (model.b2, db2, vb2),
                (model.W3, dW3, vW3), (model.b3, db3, vb3),
            ]:
                v[...] = momentum * v + lr * grad
                param -= v

        if (epoch + 1) % 20 == 0:
            avg = total_loss / n
            acc = sum(
                model.predict(features[i]) == labels[i] for i in range(n)
            ) / n
            print(f"  epoch {epoch+1:4d}/{epochs}  loss={avg:.4f}  acc={acc:.3f}")

    return model


# ---------------------------------------------------------------------------
# Kerasify export using opm-common (if available) or custom binary format
# ---------------------------------------------------------------------------

def export_opm_kerasify(model: MLP, out_path: str):
    """Export via opm-common's kerasify.export_model()."""
    if not HAS_OPM_ML:
        raise ImportError(
            "opm.ml not found.  Install opm-common's Python package or use "
            "--format=binary to write the custom format instead."
        )

    layers = [
        Dense(14, 32, "relu"),
        Dense(32, 16, "relu"),
        Dense(16, 24, "linear"),
    ]
    # Inject trained weights into the numpy Dense layers
    layers[0].weights = model.W1.T   # Dense stores weights as (in, out)
    layers[0].biases  = model.b1
    layers[1].weights = model.W2.T
    layers[1].biases  = model.b2
    layers[2].weights = model.W3.T
    layers[2].biases  = model.b3

    seq = Sequential(layers)
    export_model(seq, out_path)
    print(f"Exported Kerasify model → {out_path}")


def export_binary_fallback(model: MLP, out_path: str):
    """Write a self-contained binary that export_opm_kerasify() would produce.

    Format (same as what kerasify.py writes for 3 Dense layers):
        uint32  num_layers = 3
        --- layer 0: Dense(14→32, relu) ---
        uint32  type = 3  (kDense)
        uint32  rows = 14  (input_dim)
        uint32  cols = 32  (output_dim)
        uint32  bias_shape = 32
        float32[14*32]  weights  (row-major: W[out, in] → stored in[in, out] order)
        float32[32]     biases
        uint32  activation = 2  (kRelu)
        --- layer 1: Dense(32→16, relu) ---  ...
        --- layer 2: Dense(16→24, linear) --- activation = 1 (kLinear)
    """
    LAYER_DENSE    = 3
    ACT_LINEAR     = 1
    ACT_RELU       = 2

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", 3))  # num_layers

        for (W, b), act in zip(
            [(model.W1, model.b1), (model.W2, model.b2), (model.W3, model.b3)],
            [ACT_RELU, ACT_RELU, ACT_LINEAR],
        ):
            out_dim, in_dim = W.shape
            f.write(struct.pack("<I", LAYER_DENSE))
            f.write(struct.pack("<I", in_dim))   # rows
            f.write(struct.pack("<I", out_dim))  # cols
            f.write(struct.pack("<I", out_dim))  # bias_shape
            # weights stored column-major (in, out) to match kerasify convention
            f.write(W.T.astype(np.float32).tobytes())
            f.write(b.astype(np.float32).tobytes())
            f.write(struct.pack("<I", act))

    print(f"Exported binary model → {out_path}")


# ---------------------------------------------------------------------------
# Data collection instructions
# ---------------------------------------------------------------------------

DATA_COLLECTION_HELP = """
Data collection
===============

Run the OPM Flow simulator with each of the four standard configurations
across a set of representative benchmark decks.  Record the per-Newton-step
wall time for each (features, action) pair.

  for conf in cprw cpr_trueimpes cpr_quasiimpes cpr_trueimpesanalytic; do
    OPM_LOG_LEVEL=debug \\
      flow --linear-solver=$conf \\
           --cpr-reuse-setup=0 \\
           --output-dir=sweep/$conf \\
      DECK.DATA 2>sweep/$conf/debug.log
  done

Extract features from the debug log (NeuralCprPolicy: lines) and wall times
from the summary.  The label for each sample is the action index (0–23) whose
wall-time-per-linear-solve was minimum across the four configurations:

    python collect_labels.py --sweep-dir sweep/ --out sweep_data.csv

The CSV must have columns:
  dt_days, dt_ratio, nl_residual_norm, nl_residual_reduce, nl_iteration,
  nnz_per_row, diag_dominance, prev_linsolver_iters, prev_solve_failed,
  time_elapsed_frac, num_cells_log, block_size,
  best_weight_type, best_use_cprw, best_fine_smoother, best_coarse_smoother

Recommended decks: SPE1, SPE3, SPE9, NORNE_ATW2013
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_csv(path: str) -> tuple[list[np.ndarray], list[int]]:
    features, labels = [], []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            features.append(normalise(row))
            labels.append(action_index(
                row["best_weight_type"],
                row["best_use_cprw"],
                row["best_fine_smoother"],
                row["best_coarse_smoother"],
            ))
    return features, labels


def main():
    parser = argparse.ArgumentParser(
        description="Train and export the NeuralCprPolicy MLP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DATA_COLLECTION_HELP,
    )
    parser.add_argument("--data",   required=True,
                        help="CSV file produced by the sweep (see --help for format)")
    parser.add_argument("--out",    default="cpr_policy.model",
                        help="Output model file (default: cpr_policy.model)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr",     type=float, default=0.01)
    parser.add_argument("--format", choices=["kerasify", "binary"], default="kerasify",
                        help="kerasify = use opm-common export (default); "
                             "binary = standalone fallback")
    args = parser.parse_args()

    print(f"Loading data from {args.data} ...")
    features, labels = load_csv(args.data)
    print(f"  {len(features)} samples, "
          f"{len(set(labels))} distinct actions out of 24")

    print(f"Training for {args.epochs} epochs ...")
    model = train(features, labels, epochs=args.epochs, lr=args.lr)

    if args.format == "kerasify":
        export_opm_kerasify(model, args.out)
    else:
        export_binary_fallback(model, args.out)

    # Quick sanity check: accuracy on training set
    correct = sum(model.predict(f) == l for f, l in zip(features, labels))
    print(f"Training accuracy: {correct}/{len(features)} "
          f"({100*correct/len(features):.1f}%)")
    print(f"\nTo use the model:\n"
          f"  export OPM_NEURAL_CPR_WEIGHTS={os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
