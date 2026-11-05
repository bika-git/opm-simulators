#!/usr/bin/env python3
"""Evaluate an already exported .model file (binary NNModel format) without
retraining.

  policy  neural CPR-variant classifier (train_cpr_policy.py):
          accuracy and top-3 accuracy, overall, on the validation split the
          trainer used (same --val-frac/--seed) and per deck.
  amg     neural AMG cost model (train_cpr_amg_policy.py):
          validation MSE on the trainer's held-out report steps and the
          anchored-rule check per deck.
  abort   neural abort classifier (train_abort_policy.py):
          AUC and net saving per threshold on the given DBG runs.

Examples
  python3 eval_model.py policy MODEL --data cpr_training_data.csv
  python3 eval_model.py amg    MODEL --csvs sweep_amg_*/*_pd*_ct*.csv
  python3 eval_model.py abort  MODEL --runs label_*/SPE11A.DBG
"""
import argparse
import math
import struct
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import continual  # noqa: F401
except ModuleNotFoundError:
    # The trainers import continual.py (EWC/replay) at module level; it is not
    # in the tree, and evaluation does not need it.
    sys.modules["continual"] = types.SimpleNamespace(**{n: None for n in (
        "compute_fisher_diagonal", "ewc_penalty_grads", "save_anchor", "load_anchor",
        "merge_fisher", "save_replay_buffer", "load_replay_buffer", "select_exemplars")})

ACT = {1: "linear", 2: "relu", 3: "softplus", 6: "tanh"}


def read_model(path):
    """Read the format written by train_cpr_policy.export_binary_fallback."""
    raw = open(path, "rb").read()
    off = 0

    def u32():
        nonlocal off
        v = struct.unpack_from("<I", raw, off)[0]
        off += 4
        return v

    def f32(n):
        nonlocal off
        a = np.frombuffer(raw, "<f4", n, off)
        off += 4 * n
        return a

    layers = []
    for _ in range(u32()):
        assert u32() == 3, "only dense layers are supported"
        n_in, n_out, _ = u32(), u32(), u32()
        W = f32(n_in * n_out).reshape(n_in, n_out)
        b = f32(n_out)
        layers.append((W, b, ACT[u32()]))
    return layers


def predict(layers, X):
    h = np.asarray(X, np.float64)
    for W, b, act in layers:
        h = h @ W + b
        if act == "relu":
            h = np.maximum(h, 0)
        elif act == "tanh":
            h = np.tanh(h)
        elif act == "softplus":
            h = np.log1p(np.exp(h))
    return h


def describe(layers):
    dims = [layers[0][0].shape[0]] + [W.shape[1] for W, _, _ in layers]
    n = sum(W.size + b.size for W, b, _ in layers)
    print(f"Model: {'→'.join(map(str, dims))}, {n} parameters")


def topk_acc(out, y, k):
    return float(np.mean([t in np.argsort(-o)[:k] for o, t in zip(out, y)]))


def eval_policy(layers, args):
    from train_cpr_policy import load_csv
    feats, labels, decks = load_csv(args.data)
    X, y, decks = np.array(feats), np.array(labels), np.array(decks)
    out = predict(layers, X)
    perm = np.random.default_rng(args.seed).permutation(len(y))
    val = np.zeros(len(y), bool)
    val[perm[:max(1, int(len(y) * args.val_frac))]] = True

    def line(tag, m):
        print(f"  {tag:24s} n={m.sum():6d}  acc={topk_acc(out[m], y[m], 1):.3f}  "
              f"top3={topk_acc(out[m], y[m], 3):.3f}")
    print(f"{len(y)} samples, {len(set(labels))} labels")
    line("all", np.ones(len(y), bool))
    line("train split", ~val)
    line(f"val split (seed {args.seed})", val)
    for d in sorted(set(decks)):
        line(d, decks == d)
    print(f"Majority-class acc: {Counter(labels).most_common(1)[0][1] / len(y):.3f}")


def eval_amg(layers, args):
    from train_cpr_amg_policy import DAMPING, TARGET, load_windows, slots
    X, di, ti, y, deck, grp = load_windows(args.csvs)
    out = -predict(layers, X)                    # exported as -log cost
    additive = out.shape[1] == len(DAMPING) + len(TARGET)
    names, inv, counts = np.unique(deck, return_inverse=True, return_counts=True)
    w = (len(deck) / (len(names) * counts))[inv]
    rng = np.random.default_rng(args.seed)
    groups = sorted(set(grp))
    val = set()
    for d in names:
        gd = [g for g in groups if g[0] == d]
        val.update(gd[i] for i in rng.choice(len(gd), max(1, round(len(gd) * args.val_frac)),
                                              replace=False))
    is_val = np.array([g in val for g in grp])
    pred = np.array([sum(out[i, k] for k in slots(di[i], ti[i], additive))
                     for i in range(len(y))])
    se = (pred - y) ** 2
    for tag, m in (("train split", ~is_val), (f"val split (seed {args.seed})", is_val)):
        mse = float(np.average(se[m], weights=w[m]))
        print(f"  {tag:24s} n={m.sum():5d}  mse={mse:.4f} "
              f"(≈ ±{100 * (math.exp(math.sqrt(mse)) - 1):.0f}% per window)")
    for d in names:
        m = is_val & (deck == d)
        if m.any():
            print(f"    {d:20s} val n={m.sum():5d}  mse={float(np.mean(se[m])):.4f}")
    combos = [(d, t) for d in range(3) for t in range(3)]
    for dk in names:
        on = deck == dk
        price = {c: float(np.mean(y[on & (di == c[0]) & (ti == c[1])])) for c in combos
                 if (on & (di == c[0]) & (ti == c[1])).any()}
        states = on & is_val & (di == 0) & (ti == 0)
        if (0, 0) not in price or not states.any():
            continue
        rel = {c: math.exp(price[c] - price[(0, 0)]) - 1 for c in price}
        best = min(price, key=price.get)
        score = np.stack([sum(out[states, k] for k in slots(d, t, additive))
                          for d, t in combos], 1)
        score[:, ~np.array([c in price for c in combos])] = np.inf
        gain = score[:, 0] - score.min(1)
        print(f"\n=== {dk}: {states.sum()} held-out default-run windows; best fixed "
              f"pd{DAMPING[best[0]]}/ct{TARGET[best[1]]} {100 * rel[best]:+.1f}%")
        for m in args.margins:
            pick = np.where(gain > m, score.argmin(1), 0)
            cost = np.mean([rel[combos[k]] for k in pick])
            top = Counter(f"pd{DAMPING[combos[k][0]]}/ct{TARGET[combos[k][1]]}" for k in pick)
            print(f"  margin {m:4.2f}: priced cost {100 * cost:+5.1f}%  picks "
                  f"{dict(top.most_common(3))}")


def eval_abort(layers, args):
    from abort_feasibility import dataset, evaluate
    for run in dataset(args.runs, "solver"):
        total = sum(len(a["mb"]) for a in run["att"])
        p = 1 / (1 + np.exp(-predict(layers, run["X"])[:, 0]))
        print(f"\n=== {run['path']}: {len(run['att'])} attempts, "
              f"{sum(not a['ok'] for a in run['att'])} chopped")
        if run["y"].min() != run["y"].max():
            pos, neg = p[run["y"] == 1], p[run["y"] == 0]
            print(f"  AUC {(pos[:, None] > neg[None, :]).mean():.3f}")
        for th in args.thresholds:
            s, c15, tp, fp = evaluate(run, p > th, 1.5)
            _, c3, _, _ = evaluate(run, p > th, 3.0)
            print(f"  threshold {th}: aborts {tp} doomed, {fp} good; net saving "
                  f"{100 * (s - c15) / total:+.1f}% (redo x1.5), "
                  f"{100 * (s - c3) / total:+.1f}% (redo x3) of linearisations")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    p = sub.add_parser("policy")
    p.add_argument("--data", required=True)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("amg")
    p.add_argument("--csvs", nargs="+", required=True)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1])
    p = sub.add_parser("abort")
    p.add_argument("--runs", nargs="+", required=True,
                   help="DBG files, each with its sweep CSV next to it")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.9, 0.95, 0.98])
    for p in sub.choices.values():
        p.add_argument("model")
    args = ap.parse_args()

    layers = read_model(args.model)
    describe(layers)
    {"policy": eval_policy, "amg": eval_amg, "abort": eval_abort}[args.kind](layers, args)


if __name__ == "__main__":
    main()
