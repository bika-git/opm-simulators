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

Continual learning
------------------
When training on new decks incrementally, use --anchor to load the previous
model's anchor weights and Fisher diagonal, and --replay-buffer to mix in
exemplars from old training data.  The EWC penalty prevents catastrophic
forgetting; the replay buffer keeps the old decision boundary alive.

Run flow with
    OPM_CPR_ABORT_MODEL=abort.model OPM_CPR_ABORT_THRESHOLD=0.95 flow ...
"""
import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abort_feasibility import (CPR_FEATURES, RES_FEATURES, SOLVER_KNOWN, TinyMLP,  # noqa: E402
                               dataset, evaluate)
from train_cpr_policy import export_binary_fallback  # noqa: E402
from continual import (compute_fisher_diagonal, ewc_penalty_grads,  # noqa: E402
                       save_anchor, load_anchor, merge_fisher,
                       save_replay_buffer, load_replay_buffer,
                       select_exemplars)

FEATURES = SOLVER_KNOWN + CPR_FEATURES + RES_FEATURES   # order used by flow


def fit_continual(X, y, hidden=16, epochs=60, lr=3e-3, batch=256, seed=0,
                  ewc_anchor=None, ewc_fisher=None, ewc_lambda=1.0):
    """TinyMLP training with EWC support.

    Reproduces TinyMLP.fit() but adds the EWC gradient penalty when an
    anchor and Fisher diagonal are provided.
    """
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    d = Z.shape[1]
    P = [rng.standard_normal((d, hidden)) * math.sqrt(2 / d), np.zeros(hidden),
         rng.standard_normal(hidden) * math.sqrt(1 / hidden), np.zeros(1)]
    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]
    t = 0
    pnames = ["W0", "b0", "W1", "b1"]  # match continual.py key convention
    for _ in range(epochs):
        perm = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            b = perm[s:s + batch]
            h = np.maximum(Z[b] @ P[0] + P[1], 0)
            p = 1 / (1 + np.exp(-(h @ P[2] + P[3][0])))
            g = (p - y[b]) / len(b)
            dh = np.outer(g, P[2]) * (h > 0)
            grads = [Z[b].T @ dh, dh.sum(0), h.T @ g, np.array([g.sum()])]
            # EWC penalty
            if ewc_anchor is not None and ewc_lambda > 0:
                params_d = {pnames[k]: P[k] for k in range(4)}
                ewc_g = ewc_penalty_grads(params_d, ewc_anchor, ewc_fisher,
                                          ewc_lambda)
                for k in range(4):
                    grads[k] = grads[k] + ewc_g[pnames[k]]
            t += 1
            for k in range(4):
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    model = TinyMLP()
    model.mu, model.sd, model.P = mu, sd, P
    return model


def export(model, path):
    """Fold (x - mu) / sd into the first layer and write the binary model."""
    P0 = model.P[0] / model.sd[:, None]           # (features, hidden)
    b0 = model.P[1] - (model.mu / model.sd) @ model.P[0]
    net = SimpleNamespace(activation="relu",
                          weights=[P0.T.astype(np.float32), model.P[2][None, :].astype(np.float32)],
                          biases=[b0.astype(np.float32), model.P[3].astype(np.float32)])
    export_binary_fallback(net, path)


def _model_params(model):
    """Extract named parameters from a TinyMLP for Fisher/anchor use."""
    return {"W0": model.P[0], "b0": model.P[1],
            "W1": model.P[2], "b1": model.P[3]}


def _model_grad_fn(model):
    """Return a per-sample gradient function for Fisher estimation."""
    Z_cache = None

    def grad_fn(x_row, y_val):
        nonlocal Z_cache
        z = (x_row - model.mu) / model.sd
        h = np.maximum(z @ model.P[0] + model.P[1], 0)
        p = 1 / (1 + np.exp(-(h @ model.P[2] + model.P[3][0])))
        g = p - y_val
        dh = (g * model.P[2]) * (h > 0)
        return {"W0": np.outer(z, dh), "b0": dh,
                "W1": h * g,           "b1": np.array([g])}
    return grad_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True, help="DBG files (sweep runs) to train on")
    ap.add_argument("--test", nargs="*", default=[], help="DBG files to evaluate on")
    ap.add_argument("--out", required=True, help="Output model file")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)

    # ── Continual learning ──
    cl = ap.add_argument_group("continual learning")
    cl.add_argument("--anchor", default=None,
                    help="Anchor .npz from previous training (enables EWC)")
    cl.add_argument("--ewc-lambda", type=float, default=1.0,
                    help="EWC penalty strength (default 1.0)")
    cl.add_argument("--replay-buffer", default=None,
                    help="Replay buffer .npz — old samples mixed into training")
    cl.add_argument("--replay-frac", type=float, default=0.5,
                    help="Replay samples as fraction of new data (default 0.5)")
    cl.add_argument("--save-anchor", default=None,
                    help="Save anchor + Fisher to .npz after training")
    cl.add_argument("--save-replay", default=None,
                    help="Save replay buffer to .npz after training")
    cl.add_argument("--replay-budget", type=int, default=500,
                    help="Exemplar budget per source for replay (default 500)")
    cl.add_argument("--fisher-decay", type=float, default=0.9,
                    help="EMA decay for merging Fisher diagonals (default 0.9)")
    args = ap.parse_args()

    train_runs = dataset(args.train, "solver")
    X = np.vstack([r["X"] for r in train_runs])
    y = np.concatenate([r["y"] for r in train_runs])
    assert X.shape[1] == len(FEATURES), (X.shape, len(FEATURES))
    print(f"train: {len(train_runs)} runs, {len(y)} decision points, "
          f"{100 * y.mean():.1f}% in attempts that get chopped")

    # ── Replay: mix old exemplars into training data ──
    if args.replay_buffer:
        replay_f, replay_l, _ = load_replay_buffer(args.replay_buffer)
        n_replay = min(int(len(y) * args.replay_frac), len(replay_f))
        rng_r = np.random.default_rng(args.seed)
        r_idx = rng_r.choice(len(replay_f), n_replay, replace=False)
        X_replay = np.array([replay_f[i] for i in r_idx], dtype=np.float64)
        y_replay = np.array([replay_l[i] for i in r_idx], dtype=np.float64)
        X = np.vstack([X, X_replay])
        y = np.concatenate([y, y_replay])
        print(f"  Replay: mixed in {n_replay} samples from {args.replay_buffer}")

    # ── EWC anchor ──
    ewc_anchor, ewc_fisher = None, None
    if args.anchor:
        ewc_anchor, ewc_fisher, anchor_decks = load_anchor(args.anchor)
        print(f"  EWC anchor: {args.anchor} (λ={args.ewc_lambda}"
              + (f", decks={anchor_decks}" if anchor_decks else "")
              + ")")

    # ── Train ──
    if ewc_anchor is not None:
        model = fit_continual(X, y, hidden=args.hidden, epochs=args.epochs,
                              seed=args.seed, ewc_anchor=ewc_anchor,
                              ewc_fisher=ewc_fisher,
                              ewc_lambda=args.ewc_lambda)
    else:
        model = TinyMLP().fit(X, y, hidden=args.hidden, epochs=args.epochs,
                              seed=args.seed)
    export(model, args.out)

    # Check that the exported (folded) network reproduces the trained one.
    W0, b0 = model.P[0] / model.sd[:, None], model.P[1] - (model.mu / model.sd) @ model.P[0]
    folded = 1 / (1 + np.exp(-(np.maximum(X[:1000] @ W0 + b0, 0) @ model.P[2] + model.P[3][0])))
    print(f"folded export max |diff| on 1000 points: "
          f"{np.abs(folded - model.prob(X[:1000])).max():.2e}")

    # ── Save anchor + Fisher ──
    if args.save_anchor:
        grad_fn = _model_grad_fn(model)
        feats = [X[i] for i in range(len(X))]
        labs = y.tolist()
        new_fisher = compute_fisher_diagonal(
            _model_params(model), grad_fn, feats, labs, seed=args.seed)
        if ewc_fisher is not None:
            merged = merge_fisher(ewc_fisher, new_fisher, decay=args.fisher_decay)
        else:
            merged = new_fisher
        sources = [str(Path(p).stem) for p in args.train]
        save_anchor(_model_params(model), merged, args.save_anchor,
                    decks=sources)
        print(f"  Anchor + Fisher → {args.save_anchor}")

    # ── Save replay buffer ──
    if args.save_replay:
        feats = [X[i].astype(np.float32) for i in range(len(X))]
        labs = y.tolist()
        # No deck stratification for abort (stratify uniformly)
        sel_f, sel_l, _ = select_exemplars(
            feats, labs, budget_per_deck=args.replay_budget, seed=args.seed)
        save_replay_buffer(sel_f, sel_l, args.save_replay)
        print(f"  Replay buffer → {args.save_replay} ({len(sel_f)} exemplars)")

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
