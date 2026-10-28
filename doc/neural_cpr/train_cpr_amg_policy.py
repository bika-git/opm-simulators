#!/usr/bin/env python3
"""Train a cost model for the CPR pressure AMG parameters.

Actions: prolongation damping (1.0 / 1.3 / 1.6) and coarsening target
(1200 / 600 / 300), chosen at CPR setup rebuilds.  By default one output per
combination (3 x 3, damping major), which captures interactions (on SPE11A a
coarsening target of 300 is slower with damping 1.0 but fastest with 1.6).
--additive instead uses one output per parameter level (3 + 3),
    log cost(damping d, target t) ≈ out[d] + out[3 + t],
which grows with the sum of levels rather than their product.

Samples are CPR reuse windows (a setup rebuild plus the solves reusing it)
with denoised cost (train_cpr_cost.modelled_ms), from a sweep whose CSVs are
named <deck>_pd<damping>_ct<target>.csv (tmp/sweep_amg_*/driver.py).

Export: outputs are negated (-log cost), so higher is cheaper; flow
(OPM_NEURAL_CPR_ACTIONS=amg) recognises the layout by the number of outputs.
The default (1.0, 1200) is the anchor.
"""
import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cpr_cost import forward, iter_rows, modelled_ms, pack_ensemble  # noqa: E402
from train_cpr_policy import MLP, normalise, export_binary_fallback  # noqa: E402
from continual import (save_anchor, load_anchor, merge_fisher,  # noqa: E402
                       save_replay_buffer, load_replay_buffer)

DAMPING = [1.0, 1.3, 1.6]
TARGET = [1200, 600, 300]
N_FEAT = 12
TAG = re.compile(r"_pd([0-9.]+)_ct(\d+)\.csv$")


def slots(d, t, additive):
    """Output indices whose sum is the predicted log cost of combination (d, t)."""
    if additive:
        return [d, len(DAMPING) + t]
    return [d * len(TARGET) + t]


def load_windows(paths):
    """X (n x 12), damping index, target index, log cost per solve, deck,
    report-step group, and whether the window comes from the default run."""
    by_src = defaultdict(list)
    for _, src, row in iter_rows(paths):
        by_src[src].append(row)
    X, di, ti, y, deck_of, grp = [], [], [], [], [], []
    for src, rows in by_src.items():
        m = TAG.search(src)
        deck = src[:m.start()]
        d, t = DAMPING.index(float(m[1])), TARGET.index(int(m[2]))
        ms = modelled_ms(rows)
        start = None
        for i, r in enumerate(rows + [None]):
            if r is None or r["setup_rebuilt"] == "1":
                if start is not None:
                    X.append(normalise(rows[start])[:N_FEAT])
                    di.append(d)
                    ti.append(t)
                    y.append(math.log(max(ms[start:i].mean(), 1e-3)))
                    deck_of.append(deck)
                    grp.append((deck, rows[start]["episode"]))
                start = i
    return (np.asarray(X, np.float32), np.asarray(di), np.asarray(ti),
            np.asarray(y, np.float32), np.asarray(deck_of), grp)


def train(X, di, ti, y, w, Xv, dv, tv, yv, wv, hidden, depth, epochs, lr, batch,
          seed, patience, additive,
          ewc_anchor=None, ewc_fisher=None, ewc_lambda=0.0):
    n_out = len(DAMPING) + len(TARGET) if additive else len(DAMPING) * len(TARGET)
    mlp = MLP(hidden=hidden, activation="relu", seed=seed, n_features=N_FEAT, depth=depth)
    mlp.weights[-1] = mlp.weights[-1][:n_out].copy()
    mlp.biases[-1] = np.zeros(n_out, np.float32)
    Ws, bs = mlp.weights, mlp.biases
    if additive:
        bs[-1][0] = y.mean()                   # start at the mean cost
    else:
        for d in range(len(DAMPING)):          # start each combination at its mean
            for t in range(len(TARGET)):
                sel = (di == d) & (ti == t)
                bs[-1][slots(d, t, False)[0]] = y[sel].mean() if sel.any() else y.mean()
    params = Ws + bs
    pnames = [f"W{i}" for i in range(len(Ws))] + [f"b{i}" for i in range(len(bs))]
    m_ = [np.zeros_like(p) for p in params]
    v_ = [np.zeros_like(p) for p in params]
    rng = np.random.default_rng(seed)
    best, best_state, bad, t = np.inf, None, 0, 0

    def cols(d, tt):
        return [np.array([slots(a, b, additive)[k] for a, b in zip(d, tt)])
                for k in range(len(slots(0, 0, additive)))]

    def pred(out, d, tt):
        return sum(out[np.arange(len(d)), c] for c in cols(d, tt))

    for ep in range(epochs):
        lr_ep = 1e-5 + 0.5 * (lr - 1e-5) * (1 + math.cos(math.pi * ep / epochs))
        perm = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            b = perm[s:s + batch]
            out, hs = forward(Ws, bs, X[b])
            err = pred(out, di[b], ti[b]) - y[b]
            g = np.zeros_like(out)
            for c in cols(di[b], ti[b]):
                g[np.arange(len(b)), c] += 2 * w[b] * err / len(b)
            gW, gb = [None] * len(Ws), [None] * len(bs)
            for i in range(len(Ws) - 1, -1, -1):
                gW[i], gb[i] = g.T @ hs[i], g.sum(0)
                if i > 0:
                    g = (g @ Ws[i]) * (hs[i] > 0)
            # EWC penalty: add λ * F_i * (θ_i - θ*_i) to each gradient
            all_grads = gW + gb
            if ewc_anchor is not None and ewc_lambda > 0:
                for j, name in enumerate(pnames):
                    if name in ewc_anchor and name in ewc_fisher:
                        all_grads[j] = all_grads[j] + ewc_lambda * ewc_fisher[name] * (params[j] - ewc_anchor[name])
            t += 1
            for j, (p, gr) in enumerate(zip(params, all_grads)):
                m_[j] = 0.9 * m_[j] + 0.1 * gr
                v_[j] = 0.999 * v_[j] + 0.001 * gr * gr
                p -= lr_ep * ((m_[j] / (1 - 0.9 ** t)) / (np.sqrt(v_[j] / (1 - 0.999 ** t)) + 1e-8)
                              + (1e-4 * p if j < len(Ws) else 0))
        out, _ = forward(Ws, bs, Xv)
        vl = float(np.average((pred(out, dv, tv) - yv) ** 2, weights=wv))
        if vl < best - 1e-5:
            best, bad, best_state = vl, 0, [p.copy() for p in params]
        else:
            bad += 1
        if bad >= patience:
            break
    for p, s in zip(params, best_state):
        p[...] = s
    print(f"  val_mse={best:.4f} (≈ ±{100 * (math.exp(math.sqrt(best)) - 1):.0f}% per window)")
    return mlp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1])
    ap.add_argument("--additive", action="store_true",
                    help="One output per parameter level instead of per combination")
    ap.add_argument("--seed", type=int, default=42)

    # ── Continual learning ──
    cl = ap.add_argument_group("continual learning")
    cl.add_argument("--anchor", default=None,
                    help="Anchor .npz from previous training (enables EWC)")
    cl.add_argument("--ewc-lambda", type=float, default=1.0,
                    help="EWC penalty strength (default 1.0)")
    cl.add_argument("--replay-buffer", default=None,
                    help="Replay buffer .npz — old windows mixed into training")
    cl.add_argument("--replay-frac", type=float, default=0.5,
                    help="Replay windows as fraction of new data (default 0.5)")
    cl.add_argument("--save-anchor", default=None,
                    help="Save anchor + Fisher to .npz after training")
    cl.add_argument("--save-replay", default=None,
                    help="Save replay buffer to .npz after training")
    cl.add_argument("--fisher-decay", type=float, default=0.9,
                    help="EMA decay for merging Fisher diagonals (default 0.9)")
    args = ap.parse_args()

    X, di, ti, y, deck, grp = load_windows(args.csvs)
    print(f"{len(y)} reuse windows from {len(args.csvs)} runs: "
          f"{dict(Counter(deck.tolist()))}")

    # ── Replay: mix old windows into training data ──
    if args.replay_buffer:
        rdata = np.load(args.replay_buffer, allow_pickle=True)
        n_avail = len(rdata["y"])
        n_replay = min(int(len(y) * args.replay_frac), n_avail)
        rng_r = np.random.default_rng(args.seed)
        r_idx = rng_r.choice(n_avail, n_replay, replace=False)
        X = np.vstack([X, rdata["X"][r_idx]])
        di = np.concatenate([di, rdata["di"][r_idx]])
        ti = np.concatenate([ti, rdata["ti"][r_idx]])
        y = np.concatenate([y, rdata["y"][r_idx]])
        deck = np.concatenate([deck, rdata["deck"][r_idx]])
        grp = grp + [tuple(g) for g in rdata["grp"][r_idx]]
        print(f"  Replay: mixed in {n_replay} windows from {args.replay_buffer}")

    # ── EWC anchor ──
    ewc_anchor, ewc_fisher = None, None
    if args.anchor:
        ewc_anchor, ewc_fisher, anchor_decks = load_anchor(args.anchor)
        print(f"  EWC anchor: {args.anchor} (λ={args.ewc_lambda}"
              + (f", decks={anchor_decks}" if anchor_decks else "")
              + ")")

    names, inv, counts = np.unique(deck, return_inverse=True, return_counts=True)
    w = (len(deck) / (len(names) * counts))[inv].astype(np.float32)   # decks weigh equally

    rng = np.random.default_rng(args.seed)
    groups = sorted(set(grp))
    val = set()
    for d in names:
        gd = [g for g in groups if g[0] == d]
        val.update(gd[i] for i in rng.choice(len(gd), max(1, round(len(gd) * args.val_frac)),
                                              replace=False))
    is_val = np.array([g in val for g in grp])
    tr = np.flatnonzero(~is_val)
    members = []
    for j in range(args.ensemble):
        idx = tr if j == 0 else np.random.default_rng(args.seed + j).choice(tr, len(tr))
        members.append(train(X[idx], di[idx], ti[idx], y[idx], w[idx], X[is_val], di[is_val],
                             ti[is_val], y[is_val], w[is_val], args.hidden, args.depth,
                             args.epochs, args.lr, args.batch, args.seed + j, args.patience,
                             args.additive,
                             ewc_anchor=ewc_anchor, ewc_fisher=ewc_fisher,
                             ewc_lambda=args.ewc_lambda if ewc_anchor else 0.0))
    mlp = members[0] if args.ensemble == 1 else pack_ensemble(members, args.hidden, args.depth)

    # Offline check of the anchored rule on held-out windows of the default
    # run, each pick priced at its combination's mean cost on the deck.
    out, _ = forward(mlp.weights, mlp.biases, X)
    combos = [(d, t) for d in range(3) for t in range(3)]
    for dk in names:
        on = deck == dk
        price = {}
        for d, t in combos:
            sel = on & (di == d) & (ti == t)
            if sel.any():
                price[(d, t)] = float(np.mean(y[sel]))
        if (0, 0) not in price:
            continue
        rel = {c: math.exp(price[c] - price[(0, 0)]) - 1 for c in price}
        best = min(price, key=price.get)
        states = on & is_val & (di == 0) & (ti == 0)
        if not states.any():
            continue
        score = np.stack([sum(out[states, k] for k in slots(d, t, args.additive))
                          for d, t in combos], 1)
        avail = np.array([c in price for c in combos])
        score[:, ~avail] = np.inf
        gain = score[:, 0] - score.min(1)
        print(f"\n=== {dk}: {states.sum()} held-out default-run windows; best fixed "
              f"pd{DAMPING[best[0]]}/ct{TARGET[best[1]]} {100 * rel[best]:+.1f}%")
        for m in args.margins:
            pick = np.where(gain > m, score.argmin(1), 0)
            cost = np.mean([rel[combos[k]] for k in pick])
            top = Counter(f"pd{DAMPING[combos[k][0]]}/ct{TARGET[combos[k][1]]}" for k in pick)
            print(f"  margin {m:4.2f}: priced cost {100 * cost:+5.1f}%  picks "
                  f"{dict(top.most_common(3))}")

    # ── Save anchor + Fisher from member 0 (before negation) ──
    if args.save_anchor:
        m0 = members[0]
        m0_params = {f"W{i}": m0.weights[i] for i in range(len(m0.weights))}
        m0_params.update({f"b{i}": m0.biases[i] for i in range(len(m0.biases))})
        # Approximate Fisher via MSE gradient on training windows
        n_fisher = min(1000, len(tr))
        fisher_dict: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in m0_params.items()}
        for fi in range(n_fisher):
            ii = tr[fi]
            out_1, hs_1 = forward(m0.weights, m0.biases, X[ii:ii+1])
            err_1 = float(sum(out_1[0, k] for k in slots(di[ii], ti[ii], args.additive)) - y[ii])
            g_out = np.zeros_like(out_1)
            for c_idx in slots(di[ii], ti[ii], args.additive):
                g_out[0, c_idx] += 2 * err_1
            for layer_i in range(len(m0.weights) - 1, -1, -1):
                gW_f = g_out.T @ hs_1[layer_i]
                gb_f = g_out.sum(0)
                fisher_dict[f"W{layer_i}"] += gW_f ** 2 / n_fisher
                fisher_dict[f"b{layer_i}"] += gb_f ** 2 / n_fisher
                if layer_i > 0:
                    g_out = (g_out @ m0.weights[layer_i]) * (hs_1[layer_i] > 0)
        if ewc_fisher is not None:
            fisher_dict = merge_fisher(ewc_fisher, fisher_dict, decay=args.fisher_decay)
        all_decks = sorted(set(deck.tolist()))
        save_anchor(m0_params, fisher_dict, args.save_anchor, decks=all_decks)
        print(f"  Anchor + Fisher → {args.save_anchor} (decks: {all_decks})")

    # ── Save replay buffer ──
    if args.save_replay:
        np.savez(args.save_replay,
                 X=X, di=di, ti=ti, y=y, deck=deck,
                 grp=np.array(grp, dtype=object))
        print(f"  Replay buffer → {args.save_replay} ({len(y)} windows)")

    mlp.weights[-1] *= -1.0
    mlp.biases[-1] *= -1.0
    export_binary_fallback(mlp, args.out)


if __name__ == "__main__":
    main()
