#!/usr/bin/env python3
from __future__ import annotations
"""
Train a per-config cost model for NeuralCprPolicy from raw sweep CSVs.

Instead of classifying "the best config" (which needs matching the same
linear solve across configs, and only compares whichever configs happened to
share a timestep history), this regresses the cost of every config directly:

    MLP: 12 features → H×depth → 24 outputs,   output k ≈ log(total_ms) of config k

Every sweep row is a valid training example: it trains only the output of the
config that produced it (masked MSE).  At runtime the cheapest config is chosen.

Export
------
The model is written in the same binary format as train_cpr_policy.py, so no
C++ change is needed: the last layer is scaled so that

    exported output k = -log(ms_k) / tau

and NeuralCprPolicy's argmax picks the lowest predicted cost.  Its softmax
"confidence" becomes  1 / sum_k exp(-(log ms_k - log ms_best) / tau),  i.e. how
clearly the best config is predicted to beat the rest; with the default
OPM_NEURAL_CPR_CONFIDENCE=0.6 the policy only switches when that margin is
clear (tau=0.1 ≈ a 10 % cost difference per e-fold).

Anchored decisions (recommended): export with --tau 1, so outputs are
-log(cost), and run flow with OPM_NEURAL_CPR_MIN_GAIN=g.  The policy then stays
on the default solver unless another config is predicted cheaper by more than g
(log-cost; 0.05 ≈ 5 %), and its guards rule out configs that cannot work or are
redundant on the deck.  The offline check (--margins) previews this rule.

Timing noise is of the same size as the differences between good configs, so
--denoise (exact iteration counts × per-run cost constants instead of measured
times) and --ensemble K (mean of K bootstrap members) reduce how much of that
noise the model learns.

Configs without training data get output -1e4 and are never selected.  With
--merge-cprw (decks without wells, where CPRW == CPR) cprw:* rows train the
cpr:* output and the cprw outputs are disabled.

Only the first 12 features are used (those the sweep CSV records); the C++
side feeds the first model_input_size features, so the 7 trailing features
are not seen.

Usage
-----
python train_cpr_cost.py --csvs sweep/SPE11A_*.csv sweep/SPE11B_*.csv \\
    --merge-cprw --out cpr_cost.model
"""

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_sweep_labels import config_to_label          # noqa: E402
from train_cpr_policy import MLP, normalise, export_binary_fallback  # noqa: E402

N_FEAT = 12
N_OUT = 24
DISABLED = -1e4


def iter_rows(paths):
    """Yield (deck, source, row).  A path is either a per-config sweep CSV
    named <deck>_<NN>.csv, or a combined CSV (see --dump-combined) that has
    'deck' and 'source' columns."""
    for path in paths:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            combined = "deck" in (reader.fieldnames or [])
            for row in reader:
                if combined:
                    yield row["deck"], row["source"], row
                else:
                    yield Path(path).stem[:-3], Path(path).name, row


def modelled_ms(rows):
    """Denoised per-solve cost for the rows of one source (one config run).

    Wall-clock timings of single solves are noisy (identical runs differ by
    several %), but iteration counts are exact.  Model each solve as
        (rebuild or update cost) + s0 + s1 * linear iterations
    with the source's median rebuild/update cost and a least-squares fit of
    solve_ms on iterations, then evaluate it at the solve's iteration count.
    A solve's iterations are the next row's prev_linsolver_iters."""
    its = np.array([float(r["prev_linsolver_iters"]) for r in rows[1:]]
                   + [np.nan], dtype=float)
    its[-1] = np.nanmedian(its) if len(rows) > 1 else 1.0
    rebuilt = np.array([r["setup_rebuilt"] == "1" for r in rows])
    setup = np.array([float(r["setup_ms"]) for r in rows])
    solve = np.array([float(r["solve_ms"]) for r in rows])
    rebuild_ms = np.median(setup[rebuilt]) if rebuilt.any() else np.median(setup)
    update_ms = np.median(setup[~rebuilt]) if (~rebuilt).any() else rebuild_ms
    if np.ptp(its) > 0:
        s1, s0 = np.polyfit(its, solve, 1)
    else:
        s1, s0 = 0.0, float(np.median(solve))
    s1, s0 = max(s1, 0.0), max(s0, 0.0)
    return np.where(rebuilt, rebuild_ms, update_ms) + s0 + s1 * its


def load_rows(paths, merge_decks, windows=False, denoise=False):
    """Return X (n×12), label (n), log_ms (n), group id (n), deck (n), key (n),
    the raw rows (for --dump-combined) and the source of each sample.

    merge_decks: decks on which cprw:* is treated as cpr:* (None = all).

    windows: for sweeps run with CPR setup reuse (CSV has setup_rebuilt).  The
    policy can only switch config when the setup is rebuilt, so each sample is
    a reuse window — a rebuild solve plus the solves that reuse its setup —
    with the rebuild solve's features and target log(mean total_ms per solve
    over the window), i.e. setup cost amortised over the window.

    denoise: use modelled_ms() instead of the measured total_ms."""
    X, lab, y, grp, deck_of, key_of, src_of = [], [], [], [], [], [], []
    raw = [dict(row, deck=deck, source=src) for deck, src, row in iter_rows(paths)]
    if denoise:
        if "setup_rebuilt" not in raw[0]:
            sys.exit("--denoise needs setup_rebuilt/setup_ms/solve_ms columns")
        start = 0
        for i in range(1, len(raw) + 1):
            if i == len(raw) or raw[i]["source"] != raw[start]["source"]:
                for r, ms in zip(raw[start:i], modelled_ms(raw[start:i])):
                    r["_ms"] = ms
                start = i
    win = None  # [sample index, summed ms, solve count] of the open window

    def close_window():
        if win is not None:
            y[win[0]] = math.log(max(win[1] / win[2], 1e-3))

    source = None
    for row in raw:
        deck, src = row["deck"], row["source"]
        # Rows of one source (one config run) are in simulation order.
        if src != source:
            close_window()
            win = None
            source, episode, attempt = src, None, -1
        if row["episode"] != episode:
            episode, attempt = row["episode"], -1
        if row["newton_iter"] == "0":
            attempt += 1
        ms = row["_ms"] if denoise else float(row["total_ms"])
        if windows:
            if "setup_rebuilt" not in row:
                sys.exit(f"--windows needs a setup_rebuilt column ({src})")
            if row["setup_rebuilt"] != "1":
                if win is not None:
                    win[1] += ms
                    win[2] += 1
                continue
            close_window()
            win = [len(y), ms, 1]
        l = config_to_label(row["config"])
        if merge_decks is None or deck in merge_decks:
            l &= ~4
        X.append(normalise(row)[:N_FEAT])
        lab.append(l)
        y.append(math.log(max(ms, 1e-3)))
        grp.append((deck, episode))
        deck_of.append(deck)
        # Same solve across configs (see collect_sweep_labels.py);
        # used only for the matched-solve evaluation below.
        key_of.append((deck, episode, attempt, row["dt_days"], row["newton_iter"]))
        src_of.append(src)
    close_window()
    for r in raw:
        r.pop("_ms", None)
    return (np.asarray(X, np.float32), np.asarray(lab), np.asarray(y, np.float32),
            grp, np.asarray(deck_of), key_of, raw, np.asarray(src_of))


def forward(Ws, bs, X):
    hs = [X]
    h = X
    for i, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W.T + b
        if i < len(Ws) - 1:
            h = np.maximum(h, 0.0)
        hs.append(h)
    return h, hs


def deck_weights(decks):
    """Per-row weights giving every deck the same total weight (mean 1)."""
    names, inv, counts = np.unique(decks, return_inverse=True, return_counts=True)
    w = (len(decks) / (len(names) * counts))[inv]
    return w.astype(np.float32)


def train(X, lab, y, w, Xv, labv, yv, wv, hidden, depth, epochs, lr, batch, seed,
          patience):
    mlp = MLP(hidden=hidden, activation="relu", seed=seed, n_features=N_FEAT, depth=depth)
    Ws, bs = mlp.weights, mlp.biases
    # Start every output at the mean log cost of its config.
    for k in range(N_OUT):
        m = lab == k
        bs[-1][k] = y[m].mean() if m.any() else y.mean()
    params = Ws + bs
    m_ = [np.zeros_like(p) for p in params]
    v_ = [np.zeros_like(p) for p in params]
    t = 0
    rng = np.random.default_rng(seed)
    n = len(y)
    best, best_state, bad = np.inf, None, 0

    def val_loss():
        out, _ = forward(Ws, bs, Xv)
        return float(np.average((out[np.arange(len(yv)), labv] - yv) ** 2, weights=wv))

    for ep in range(epochs):
        lr_ep = 1e-5 + 0.5 * (lr - 1e-5) * (1 + math.cos(math.pi * ep / epochs))
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            xb, lb, yb, wb = X[idx], lab[idx], y[idx], w[idx]
            out, hs = forward(Ws, bs, xb)
            err = out[np.arange(len(idx)), lb] - yb
            tot += float((wb * err ** 2).sum())
            d = np.zeros_like(out)
            d[np.arange(len(idx)), lb] = 2 * wb * err / len(idx)   # masked, weighted MSE
            gW, gb = [None] * len(Ws), [None] * len(bs)
            for i in range(len(Ws) - 1, -1, -1):
                gW[i] = d.T @ hs[i]
                gb[i] = d.sum(0)
                if i > 0:
                    d = (d @ Ws[i]) * (hs[i] > 0)
            t += 1
            for j, (p, g) in enumerate(zip(params, gW + gb)):
                m_[j] = 0.9 * m_[j] + 0.1 * g
                v_[j] = 0.999 * v_[j] + 0.001 * g * g
                mh = m_[j] / (1 - 0.9 ** t)
                vh = v_[j] / (1 - 0.999 ** t)
                wd = 1e-4 if j < len(Ws) else 0.0
                p -= lr_ep * (mh / (np.sqrt(vh) + 1e-8) + wd * p)
        vl = val_loss()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = [p.copy() for p in params]
        else:
            bad += 1
        if (ep + 1) % 10 == 0 or bad == 0 and ep < 3:
            print(f"  epoch {ep+1:4d}  lr={lr_ep:.2e}  train_mse={tot/n:.4f}  val_mse={vl:.4f}")
        if bad >= patience:
            print(f"  Early stop at epoch {ep+1}")
            break
    for p, s in zip(params, best_state):
        p[...] = s
    print(f"  Restored best checkpoint (val_mse={best:.4f}, "
          f"≈ ±{100*(math.exp(math.sqrt(best))-1):.0f}% per solve)")
    return mlp


def pack_ensemble(members, hidden, depth):
    """Pack K identical-architecture MLPs into one wider MLP whose output is
    their mean: first layers stacked, hidden layers block-diagonal, last
    layer concatenated and divided by K.  The C++ side loads it as usual."""
    K = len(members)
    Ws, bs = [], []
    for i in range(depth + 1):
        Wi = [m.weights[i] for m in members]
        bi = [m.biases[i] for m in members]
        if i == 0:
            W, b = np.vstack(Wi), np.concatenate(bi)
        elif i < depth:
            W = np.zeros((K * hidden, K * hidden), np.float32)
            for j, Wj in enumerate(Wi):
                W[j * hidden:(j + 1) * hidden, j * hidden:(j + 1) * hidden] = Wj
            b = np.concatenate(bi)
        else:
            W, b = np.hstack(Wi) / K, np.mean(bi, axis=0)
        Ws.append(W.astype(np.float32))
        bs.append(b.astype(np.float32))
    packed = MLP(hidden=K * hidden, activation="relu", n_features=N_FEAT, depth=depth)
    packed.weights, packed.biases = Ws, bs
    return packed


def policy_check(mlp, X, lab, y, deck, src_of, is_val, no_wells, margins):
    """Offline check of the anchored decision rule used at runtime
    (OPM_NEURAL_CPR_MIN_GAIN): on held-out states of the default solver's run,
    leave the anchor only for a config predicted cheaper by more than the
    margin, with the C++ guards (cprw only on decks with wells, cpr only on
    decks without).  Each pick is priced at its config's mean cost on the deck
    — a proxy that cannot credit choices that are good only in particular
    states, so the flow runs remain the real test."""
    pred, _ = forward(mlp.weights, mlp.biases, X)
    for d in sorted(set(deck)):
        anchor = 8 if d in no_wells else 12
        on_d = deck == d
        allowed = [k for k in np.unique(lab[on_d])
                   if ((k // 4) % 2 == 1) != (d in no_wells)]
        if anchor not in allowed:
            continue
        price = {k: float(np.mean(y[on_d & (lab == k)])) for k in allowed}
        rel = {k: math.exp(price[k] - price[anchor]) - 1 for k in allowed}
        best = min(allowed, key=price.get)
        states = on_d & is_val & (lab == anchor)
        if not states.any():
            continue
        p = pred[states][:, allowed]
        gain = p[:, allowed.index(anchor)] - p.min(1)
        print(f"\n=== {d}: anchored rule on {states.sum()} held-out states "
              f"(anchor {anchor}; best fixed config: label {best}, {100*rel[best]:+.1f}%)")
        for m in margins:
            picks = np.where(gain > m, np.asarray(allowed)[p.argmin(1)], anchor)
            cost = np.mean([rel[k] for k in picks])
            print(f"  margin {m:5.3f}: leaves anchor {100*np.mean(picks != anchor):5.1f}% "
                  f"of decisions; priced cost {100*cost:+5.1f}%   "
                  f"picks {dict(Counter(picks.tolist()).most_common(4))}")


def evaluate(mlp, Xv, labv, yv, keyv, active, baseline):
    """Matched-solve check on held-out data of one deck: where the model's pick
    and the baseline config were both measured on the same solve, compare real
    times.  `active` is the set of configs that ran on this deck (the runtime
    equivalent is OPM_NEURAL_CPR_FORBID_LABELS for the rest)."""
    out, _ = forward(mlp.weights, mlp.biases, Xv)
    out[:, ~active] = np.inf
    pick = out.argmin(1)

    print("Held-out per-config log-cost RMSE:")
    print("  " + "  ".join(
        f"{k}:{math.sqrt(float(np.mean((out[labv == k, k] - yv[labv == k]) ** 2))):.2f}"
        for k in np.flatnonzero(active) if (labv == k).any()))

    times = defaultdict(dict)
    picks = defaultdict(list)
    for i, key in enumerate(keyv):
        times[key].setdefault(int(labv[i]), []).append(math.exp(yv[i]))
        picks[key].append(int(pick[i]))
    ratios, oracle, hits = [], [], 0
    for key, tk in times.items():
        if baseline not in tk or len(tk) < 2:
            continue
        p = Counter(picks[key]).most_common(1)[0][0]
        med = {k: float(np.median(v)) for k, v in tk.items()}
        if p in med:
            ratios.append(med[p] / med[baseline])
            hits += 1
        oracle.append(min(med.values()) / med[baseline])
    print(f"Pick distribution (held-out rows): "
          f"{dict(Counter(pick.tolist()).most_common())}")
    if ratios:
        g = lambda a: math.exp(np.mean(np.log(a)))
        print(f"Matched solves with baseline label {baseline}: {len(oracle)}; "
              f"model's pick measured in {hits}")
        print(f"  time(pick)/time(label {baseline}):   geo-mean {g(ratios):.3f}  "
              f"(<1 = model faster)")
        print(f"  time(best)/time(label {baseline}):   geo-mean {g(oracle):.3f}  "
              f"(oracle bound)")
    return pick


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--out", default="cpr_cost.model")
    ap.add_argument("--merge-cprw", action="store_true",
                    help="Treat cprw:* as cpr:* on all decks")
    ap.add_argument("--merge-cprw-decks", nargs="+", default=[],
                    help="Treat cprw:* as cpr:* only on these decks (no wells)")
    ap.add_argument("--windows", action="store_true",
                    help="Train on CPR reuse windows (sweeps run with setup "
                         "reuse; needs the setup_rebuilt column)")
    ap.add_argument("--denoise", action="store_true",
                    help="Replace measured solve times by a per-run cost model "
                         "evaluated at the exact iteration counts")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="Train this many bootstrap members and export their "
                         "mean as one network")
    ap.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1],
                    help="Log-cost margins for the offline anchored-rule check")
    ap.add_argument("--balance-decks", action="store_true",
                    help="Weight rows so every deck contributes equally")
    ap.add_argument("--dump-combined", default=None,
                    help="Also write all input rows to one CSV (deck, source "
                         "columns added); it can be passed back via --csvs")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="Fraction of (deck, report step) groups held out")
    ap.add_argument("--tau", type=float, default=0.1,
                    help="Log-cost temperature for the exported scores")
    ap.add_argument("--baseline", type=int, default=8,
                    help="Fixed config to compare against in evaluation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    merge = None if args.merge_cprw else set(args.merge_cprw_decks)
    X, lab, y, grp, deck, key, raw, src_of = load_rows(args.csvs, merge, args.windows,
                                                       args.denoise)
    print(f"Read {len(raw)} rows from {len(args.csvs)} files → {len(y)} "
          f"{'reuse windows' if args.windows else 'samples'}")
    for d in sorted(set(deck)):
        m = deck == d
        print(f"  {d:16s} rows={m.sum():7d}  configs="
              f"{sorted(set(lab[m].tolist()))}")
    if args.dump_combined:
        cols = ["deck", "source"] + [c for c in raw[0] if c not in ("deck", "source")]
        with open(args.dump_combined, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(raw)
        print(f"Wrote combined CSV → {args.dump_combined}")
    w = deck_weights(deck) if args.balance_decks else np.ones(len(y), np.float32)

    # Hold out whole report steps so validation states are not neighbours of
    # training states.  Pick per deck so every deck is represented.
    rng = np.random.default_rng(args.seed)
    groups = sorted(set(grp))
    val_groups = set()
    for d in sorted(set(deck)):
        gd = [g for g in groups if g[0] == d]
        k = max(1, round(len(gd) * args.val_frac))
        val_groups.update(gd[i] for i in rng.choice(len(gd), k, replace=False))
    is_val = np.array([g in val_groups for g in grp])
    print(f"Held-out report steps: {sorted(val_groups)}")
    print(f"Train rows: {(~is_val).sum()}  Val rows: {is_val.sum()}")

    active = np.zeros(N_OUT, bool)
    active[np.unique(lab)] = True
    tr = np.flatnonzero(~is_val)
    members = []
    for j in range(args.ensemble):
        # Member 0 sees the training set as is; the others a bootstrap resample.
        idx = tr if j == 0 else np.random.default_rng(args.seed + j).choice(tr, len(tr))
        if args.ensemble > 1:
            print(f"Member {j + 1}/{args.ensemble}")
        members.append(train(X[idx], lab[idx], y[idx], w[idx], X[is_val], lab[is_val],
                             y[is_val], w[is_val], args.hidden, args.depth, args.epochs,
                             args.lr, args.batch, args.seed + j, args.patience))
    mlp = members[0] if args.ensemble == 1 else pack_ensemble(members, args.hidden,
                                                               args.depth)
    if args.ensemble > 1:
        out, _ = forward(mlp.weights, mlp.biases, X[is_val])
        mse = float(np.average((out[np.arange(is_val.sum()), lab[is_val]] - y[is_val]) ** 2,
                               weights=w[is_val]))
        print(f"Ensemble of {args.ensemble}: val_mse={mse:.4f} "
              f"(≈ ±{100*(math.exp(math.sqrt(mse))-1):.0f}% per solve)")
    no_wells = set(deck.tolist()) if merge is None else merge
    policy_check(mlp, X, lab, y, deck, src_of, is_val, no_wells, args.margins)
    for d in sorted(set(deck)):
        m = is_val & (deck == d)
        deck_active = np.zeros(N_OUT, bool)
        deck_active[np.unique(lab[deck == d])] = True
        print(f"\n=== {d} (held out: {m.sum()} rows)")
        evaluate(mlp, X[m], lab[m], y[m], [k for k, v in zip(key, m) if v],
                 deck_active, args.baseline)

    # Export: output_k = -log_ms_k / tau; untrained configs disabled.
    W, b = mlp.weights[-1], mlp.biases[-1]
    W *= -1.0 / args.tau
    b *= -1.0 / args.tau
    W[~active] = 0.0
    b[~active] = DISABLED
    export_binary_fallback(mlp, args.out)
    print(f"Active configs: {np.flatnonzero(active).tolist()}  tau={args.tau}")


if __name__ == "__main__":
    main()
