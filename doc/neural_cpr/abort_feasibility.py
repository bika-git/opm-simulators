#!/usr/bin/env python3
"""Offline feasibility of an early-abort policy for timestep attempts.

At every Newton iteration of an attempt that has not converged yet, a model
predicts whether the attempt will end in a chop.  Aborting a doomed attempt
early (e.g. by giving its next CPR solve a tiny iteration limit) saves the
linearisations it would still have wasted; aborting an attempt that would have
converged wastes the work so far and forces a redo with a smaller step.

Accounting, in linearisations (flow's "Wasted" unit):
  doomed attempt with n rows, aborted after row j:   saves n - (j + 1)
  good attempt with n rows, aborted after row j:     costs (j + 1) + c * n
where c * n approximates the redo (c = 1.5 central, 3 pessimistic).

Train on the non-default SPE11A configs of a sweep, test on the default
config's runs (and on decks without chops, where any abort is a false alarm).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_attempts import parse_dbg  # noqa: E402

FEATURES = ["iteration", "log_cnv", "log_mb", "cnv_vs_start", "mb_vs_start",
            "cnv_step", "mb_step", "cnv_vs_best", "frac_increases", "oscillating",
            "after_chop", "chop_streak", "dt_change", "prev_iters"]
# CPR solver signals, from the sweep CSV (one row per linear solve, written by
# OPM_CPR_SWEEP_LOG).  Only values known before the solve are used.
CPR_FEATURES = ["last_lin_its", "max_lin_its", "setup_age", "rebuilt_now"]
# Residual 2-norm trend within the attempt: what the linear solver itself can
# see (it has the right-hand side, but not flow's CNV/MB convergence report).
RES_FEATURES = ["log_res", "res_vs_start", "res_step", "res_vs_best", "res_frac_up"]
MAXITER, REUSE_INTERVAL = 20, 30


def cpr_rows(csv_path, attempts):
    """Per decision point, the CPR features of the solve that follows it, by
    walking the sweep CSV in step with the attempts.  The CSV's newton_iter
    must match the row index; a decision point without a solve (Newton
    iteration limit) gets the previous values."""
    import csv
    rows = list(csv.DictReader(open(csv_path)))
    k, age, out = 0, 0, []
    width = len(CPR_FEATURES) + len(RES_FEATURES)
    for a in attempts:
        n = len(a["mb"])
        last = n if not a["ok"] else n - 1
        its_attempt, res = [], []
        for j in range(last):
            if k < len(rows) and int(rows[k]["newton_iter"]) == j:
                r = rows[k]
                k += 1
                rebuilt = r.get("setup_rebuilt") == "1"
                age = 0 if rebuilt else age + 1
                prev_its = float(r["prev_linsolver_iters"]) if j else 0.0
                if j:
                    its_attempt.append(prev_its)
                res.append(math.log10(max(float(r["nl_residual_norm"]), 1e-30)))
                ups = sum(res[i] > res[i - 1] for i in range(1, len(res)))
                out.append([prev_its / MAXITER,
                            max(its_attempt, default=0.0) / MAXITER,
                            min(age, REUSE_INTERVAL) / REUSE_INTERVAL,
                            float(rebuilt),
                            res[-1], res[-1] - res[0],
                            res[-1] - res[-2] if len(res) > 1 else 0.0,
                            res[-1] - min(res), ups / len(res)])
            else:
                out.append(out[-1] if out else [0.0] * width)
    if k != len(rows):
        print(f"  warning: {csv_path}: matched {k} of {len(rows)} CSV rows")
    return out


def decision_points(attempts):
    """Yield (features, attempt index, row j) for every row after which the
    attempt still has to solve (all rows of chopped attempts, all but the last
    of converged ones)."""
    prev = None
    streak = 0
    for ai, a in enumerate(attempts):
        cnv = np.log10(np.maximum(np.max(a["cnv"], axis=1), 1e-12))
        mb = np.log10(np.maximum(np.max(a["mb"], axis=1), 1e-12))
        n = len(cnv)
        after_chop = prev is not None and not prev["ok"]
        dt_change = math.log10(a["dt"] / prev["dt"]) if prev else 0.0
        prev_iters = (len(prev["mb"]) - 1) / 20 if prev and prev["ok"] else 0.0
        last = n if not a["ok"] else n - 1
        for j in range(last):
            ups = sum(cnv[i] > cnv[i - 1] for i in range(1, j + 1))
            x = [j / 20, cnv[j], mb[j], cnv[j] - cnv[0], mb[j] - mb[0],
                 cnv[j] - cnv[j - 1] if j else 0.0, mb[j] - mb[j - 1] if j else 0.0,
                 cnv[j] - cnv[:j + 1].min(), ups / (j + 1),
                 float(any(r <= j for r in a["osc_rows"])),
                 float(after_chop), min(streak, 5) / 5, dt_change, prev_iters]
            yield x, ai, j
        streak = streak + 1 if not a["ok"] else 0
        prev = a


def sweep_csv(dbg_path):
    """The sweep CSV of a sweep run: <deck>/label_NN/<deck>.DBG -> <deck>_NN.csv"""
    p = Path(dbg_path)
    csv_path = p.parent.parent.parent / f"{p.stem}_{p.parent.name[-2:]}.csv"
    return csv_path if csv_path.exists() else None


# Of the Newton-log features, those the linear solver also knows (Newton
# iteration index, step-size and chop history).
SOLVER_KNOWN = ["iteration", "after_chop", "chop_streak", "dt_change", "prev_iters"]


def dataset(paths, features):
    """features: 'newton' (convergence report), 'newton+cpr' (plus CPR solver
    and residual signals) or 'solver' (only what the linear solver sees)."""
    runs = []
    for p in paths:
        att = parse_dbg(p)
        rows = list(decision_points(att))
        base = np.array([r[0] for r in rows], float)
        X = base
        if features != "newton":
            csv_path = sweep_csv(p)
            if csv_path is None:
                sys.exit(f"--features {features} needs the sweep CSV next to {p}")
            extra = np.array(cpr_rows(csv_path, att), float)
            if features == "solver":
                X = np.hstack([base[:, [FEATURES.index(f) for f in SOLVER_KNOWN]], extra])
            else:
                X = np.hstack([base, extra])
        ai = np.array([r[1] for r in rows])
        j = np.array([r[2] for r in rows])
        y = np.array([not att[i]["ok"] for i in ai], float)
        runs.append({"path": p, "att": att, "X": X, "base": base, "ai": ai, "j": j,
                     "y": y})
    return runs


class Logistic:
    def fit(self, X, y, l2=1e-3, epochs=300, lr=0.1):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.w, self.b = np.zeros(Z.shape[1]), 0.0
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
            g = p - y
            self.w -= lr * (Z.T @ g / len(y) + l2 * self.w)
            self.b -= lr * g.mean()
        return self

    def prob(self, X):
        return 1 / (1 + np.exp(-(((X - self.mu) / self.sd) @ self.w + self.b)))


class TinyMLP:
    """One hidden layer (ReLU), sigmoid output, Adam, binary cross-entropy."""
    def fit(self, X, y, hidden=16, epochs=60, lr=3e-3, batch=256, seed=0):
        rng = np.random.default_rng(seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        d = Z.shape[1]
        P = [rng.standard_normal((d, hidden)) * math.sqrt(2 / d), np.zeros(hidden),
             rng.standard_normal(hidden) * math.sqrt(1 / hidden), np.zeros(1)]
        m = [np.zeros_like(p) for p in P]
        v = [np.zeros_like(p) for p in P]
        t = 0
        for _ in range(epochs):
            perm = rng.permutation(len(y))
            for s in range(0, len(y), batch):
                b = perm[s:s + batch]
                h = np.maximum(Z[b] @ P[0] + P[1], 0)
                p = 1 / (1 + np.exp(-(h @ P[2] + P[3][0])))
                g = (p - y[b]) / len(b)
                dh = np.outer(g, P[2]) * (h > 0)
                grads = [Z[b].T @ dh, dh.sum(0), h.T @ g, np.array([g.sum()])]
                t += 1
                for k in range(4):
                    m[k] = 0.9 * m[k] + 0.1 * grads[k]
                    v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                    P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
        self.P = P
        return self

    def prob(self, X):
        Z = (X - self.mu) / self.sd
        h = np.maximum(Z @ self.P[0] + self.P[1], 0)
        return 1 / (1 + np.exp(-(h @ self.P[2] + self.P[3][0])))


def evaluate(run, fire, c):
    """Abort each attempt at its first decision point where `fire` (a boolean
    per decision point) is set.  Returns saved, cost, n_true, n_false."""
    att, ai, j = run["att"], run["ai"], run["j"]
    first = {}
    for k in np.flatnonzero(fire):
        first.setdefault(int(ai[k]), int(j[k]))
    saved = cost = tp = fp = 0
    spent = []
    for a_idx, jj in first.items():
        a = att[a_idx]
        n = len(a["mb"])
        if a["ok"]:
            cost += (jj + 1) + c * n
            fp += 1
        else:
            saved += n - (jj + 1)
            tp += 1
            spent.append((jj + 1) / n)
    evaluate.spent = float(np.mean(spent)) if spent else float("nan")
    return saved, cost, tp, fp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True, help="DBG files to train on")
    ap.add_argument("--test", nargs="+", required=True, help="DBG files to test on")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.98])
    ap.add_argument("--features", choices=["newton", "newton+cpr", "solver"],
                    default="newton",
                    help="newton: flow's convergence report; newton+cpr: plus CPR "
                         "solver and residual signals from the sweep CSVs; solver: "
                         "only what the linear solver itself can see")
    args = ap.parse_args()

    train, test = dataset(args.train, args.features), dataset(args.test, args.features)
    X = np.vstack([r["X"] for r in train])
    y = np.concatenate([r["y"] for r in train])
    print(f"train: {len(train)} runs, {len(y)} decision points, "
          f"{100 * y.mean():.1f}% in attempts that get chopped")
    models = {"logistic": Logistic().fit(X, y), "neural (16 hidden)": TinyMLP().fit(X, y)}

    for run in test:
        att = run["att"]
        total = sum(len(a["mb"]) for a in att)
        fails = [a for a in att if not a["ok"]]
        wasted = sum(len(a["mb"]) for a in fails)
        oracle = sum(len(a["mb"]) - 1 for a in fails)
        name = Path(run["path"]).parent.name + "/" + Path(run["path"]).name
        print(f"\n=== {name}: {len(att)} attempts, {len(fails)} chopped; "
              f"{wasted}/{total} linearisations wasted ({100 * wasted / total:.1f}%)")
        print(f"  oracle (abort every doomed attempt after its first row): "
              f"saves {oracle} ({100 * oracle / total:.1f}% of all linearisations)")
        if len(run["y"]) == 0:
            continue
        # Hand-written rule for comparison: abort once CNV has grown for two
        # consecutive iterations.
        cnv_step = run["base"][:, FEATURES.index("cnv_step")]
        rule = np.zeros(len(cnv_step), bool)
        for k in range(1, len(cnv_step)):
            rule[k] = (run["ai"][k] == run["ai"][k - 1] and cnv_step[k] > 0
                       and cnv_step[k - 1] > 0)
        rows = [("rule: CNV up twice", None, rule)]
        for mname, model in models.items():
            p = model.prob(run["X"])
            if run["y"].min() != run["y"].max():
                pos, neg = p[run["y"] == 1], p[run["y"] == 0]
                auc = (pos[:, None] > neg[None, :]).mean()
                print(f"  {mname}: ranking quality (AUC) {auc:.3f}")
            for th in args.thresholds:
                rows.append((mname, th, p > th))
        for label, th, fire in rows:
            _, c3, _, _ = evaluate(run, fire, 3.0)
            s, c15, tp, fp = evaluate(run, fire, 1.5)
            tag = label if th is None else f"{label} p>{th}"
            print(f"  {tag:30s} aborts {tp:4d} doomed, {fp:4d} good | net saving "
                  f"{100 * (s - c15) / total:+5.1f}% (redo x1.5)  "
                  f"{100 * (s - c3) / total:+5.1f}% (redo x3) | doomed caught after "
                  f"{100 * evaluate.spent:3.0f}% of their iterations")


if __name__ == "__main__":
    main()
