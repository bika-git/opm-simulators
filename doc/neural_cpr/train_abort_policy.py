#!/usr/bin/env python3
"""Train and export the neural early-abort policy (NeuralAbortPolicy).

Training data: flow .DBG logs (timestep attempts and whether they were
chopped) of sweep runs, joined with their sweep CSVs (OPM_CPR_SWEEP_LOG) for
the features the linear solver itself can see — see abort_feasibility.py
(--features solver) and ISTLSolver::abortFeatures(), which must stay in step.

The network is a small MLP (14 -> hidden -> 1, ReLU) predicting the logit of
"this attempt will be chopped".  Input standardisation is folded into the first
layer, so flow feeds raw features.  The model is written in the binary format
read by Opm::ML::NNModel.

Run flow with
    OPM_CPR_ABORT_MODEL=abort.model OPM_CPR_ABORT_THRESHOLD=0.95 flow ...
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abort_feasibility import (CPR_FEATURES, RES_FEATURES, SOLVER_KNOWN, TinyMLP,  # noqa: E402
                               dataset, evaluate)
from train_cpr_policy import export_binary_fallback  # noqa: E402

FEATURES = SOLVER_KNOWN + CPR_FEATURES + RES_FEATURES   # order used by flow


def export(model, path):
    """Fold (x - mu) / sd into the first layer and write the binary model."""
    P0 = model.P[0] / model.sd[:, None]           # (features, hidden)
    b0 = model.P[1] - (model.mu / model.sd) @ model.P[0]
    net = SimpleNamespace(activation="relu",
                          weights=[P0.T.astype(np.float32), model.P[2][None, :].astype(np.float32)],
                          biases=[b0.astype(np.float32), model.P[3].astype(np.float32)])
    export_binary_fallback(net, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True, help="DBG files (sweep runs) to train on")
    ap.add_argument("--test", nargs="*", default=[], help="DBG files to evaluate on")
    ap.add_argument("--out", required=True, help="Output model file")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = dataset(args.train, "solver")
    X = np.vstack([r["X"] for r in train])
    y = np.concatenate([r["y"] for r in train])
    assert X.shape[1] == len(FEATURES), (X.shape, len(FEATURES))
    print(f"train: {len(train)} runs, {len(y)} decision points, "
          f"{100 * y.mean():.1f}% in attempts that get chopped")
    model = TinyMLP().fit(X, y, hidden=args.hidden, epochs=args.epochs, seed=args.seed)
    export(model, args.out)

    # Check that the exported (folded) network reproduces the trained one.
    W0, b0 = model.P[0] / model.sd[:, None], model.P[1] - (model.mu / model.sd) @ model.P[0]
    folded = 1 / (1 + np.exp(-(np.maximum(X[:1000] @ W0 + b0, 0) @ model.P[2] + model.P[3][0])))
    print(f"folded export max |diff| on 1000 points: "
          f"{np.abs(folded - model.prob(X[:1000])).max():.2e}")

    for run in dataset(args.test, "solver") if args.test else []:
        total = sum(len(a["mb"]) for a in run["att"])
        p = model.prob(run["X"])
        print(f"\n=== {run['path']}")
        for th in (0.9, 0.95, 0.98):
            s, c15, tp, fp = evaluate(run, p > th, 1.5)
            _, c3, _, _ = evaluate(run, p > th, 3.0)
            print(f"  threshold {th}: aborts {tp} doomed, {fp} good; net saving "
                  f"{100 * (s - c15) / total:+.1f}% (redo x1.5), "
                  f"{100 * (s - c3) / total:+.1f}% (redo x3) of linearisations")


if __name__ == "__main__":
    main()
