#!/usr/bin/env python3
"""Parse flow .DBG logs into timestep attempts (for an early-abort policy).

Each attempt (one "Starting time step" block) yields its Newton convergence
table (MB and CNV per phase, one row per linearisation), the outcome
(converged, or chopped) and, for chops, whether a linear solve failed.

Usage:  parse_attempts.py RUN.DBG ...   (prints a per-run summary)
"""
import re
import sys

START = re.compile(r"Starting time step \d+, stepsize ([0-9.eE+-]+) days, "
                   r"at day ([0-9.eE+-]+)/")
ROW = re.compile(r"^\s*(\d+)((?:\s+[-+0-9.]+e[-+]\d+)+)\s+\S+\s+\S+\s*$")


def parse_dbg(path):
    """Return the attempts of one run, in order.  Each attempt is a dict:
    dt, day       step size and start time (days)
    mb, cnv       per-row lists of the per-phase MB and CNV values
    ok            True if it converged, False if it was chopped
    linear_fail   True if a linear solve failed (the CPR iteration limit)
    osc_rows      row indices after which oscillation was detected
    """
    attempts, cur, n_mb = [], None, 0
    for line in open(path, errors="replace"):
        m = START.search(line)
        if m:
            cur = {"dt": float(m[1]), "day": float(m[2]), "mb": [], "cnv": [],
                   "ok": None, "linear_fail": False, "osc_rows": []}
            attempts.append(cur)
            continue
        if cur is None or cur["ok"] is not None:
            continue
        if line.startswith("Iter") and "CNV(" in line:
            n_mb = sum(c.startswith("MB(") for c in line.split())
            continue
        m = ROW.match(line)
        if m and n_mb:
            vals = [float(v) for v in m[2].split()]
            if len(vals) >= 2 * n_mb:
                cur["mb"].append(vals[:n_mb])
                cur["cnv"].append(vals[n_mb:2 * n_mb])
                continue
        if "Oscillating behavior detected" in line:
            cur["osc_rows"].append(len(cur["mb"]) - 1)
        elif "Newton its=" in line:
            cur["ok"] = True
        elif "Convergence failure for linear solver" in line:
            cur["linear_fail"] = True
        elif "Timestep chopped" in line:
            cur["ok"] = False
    return [a for a in attempts if a["mb"] and a["ok"] is not None]


def summary(attempts):
    fail = [a for a in attempts if not a["ok"]]
    rows = sum(len(a["mb"]) for a in attempts)
    wasted = sum(len(a["mb"]) for a in fail)
    return (f"{len(attempts)} attempts, {len(fail)} chopped "
            f"({sum(a['linear_fail'] for a in fail)} linear-solver failures), "
            f"{wasted}/{rows} linearisations in chopped attempts "
            f"({100 * wasted / max(rows, 1):.1f}%)")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"{p}: {summary(parse_dbg(p))}")
