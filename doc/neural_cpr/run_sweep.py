#!/usr/bin/env python3
"""
Generate all 24 CPR config JSON files and run the sweep with OPM flow.

Covers the full 3×2×2×2 action space:
  weight_type  : quasiimpes | trueimpes | trueimpesanalytic
  precond type : cpr | cprw
  fine smoother: paroverilu0 | dilu
  coarse smoother (AMG): ilu0 | dilu

Usage
-----
python run_sweep.py \\
    --flow   /path/to/flow \\
    --decks  /path/to/SPE9.DATA /path/to/NORNE_ATW2013.DATA \\
    --out    /tmp/sweep2

Outputs one CSV per (deck, config) in --out, named
  <deck_stem>_<label>.csv

Then combine with collect_sweep_labels.py:
  python collect_sweep_labels.py --csvs /tmp/sweep2/*.csv --out training_data.csv
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# All 24 configs  label = dec*8 + cprw*4 + fine*2 + coarse
# ---------------------------------------------------------------------------

WEIGHT_TYPES   = ["quasiimpes", "trueimpes", "trueimpesanalytic"]
PRECOND_TYPES  = ["cpr", "cprw"]
FINE_SMOOTHERS = ["paroverilu0", "dilu"]
COARSE_SMOOTHERS = ["ilu0", "dilu"]

ALL_CONFIGS = []
for dec, weight in enumerate(WEIGHT_TYPES):
    for cprw_idx, ptype in enumerate(PRECOND_TYPES):
        for fine_idx, fine in enumerate(FINE_SMOOTHERS):
            for coarse_idx, coarse in enumerate(COARSE_SMOOTHERS):
                label = dec * 8 + cprw_idx * 4 + fine_idx * 2 + coarse_idx
                ALL_CONFIGS.append({
                    "label":  label,
                    "name":   f"{ptype}:{weight}:{fine}:{coarse}",
                    "ptype":  ptype,
                    "weight": weight,
                    "fine":   fine,
                    "coarse": coarse,
                })

assert len(ALL_CONFIGS) == 24
assert len({c["label"] for c in ALL_CONFIGS}) == 24


# ---------------------------------------------------------------------------
# JSON config generation
# ---------------------------------------------------------------------------

AMG_DEFAULTS = {
    "type":              "amg",
    "alpha":             0.333333333333,
    "relaxation":        1.0,
    "iterations":        1,
    "coarsenTarget":     1200,
    "pre_smooth":        1,
    "post_smooth":       1,
    "beta":              0.0,
    "verbosity":         0,
    "maxlevel":          15,
    "skip_isolated":     0,
    "accumulate":        1,
    "prolongationdamping": 1.0,
    "maxdistance":       2,
    "maxconnectivity":   15,
    "maxaggsize":        6,
    "minaggsize":        4,
}


def make_json(cfg: dict, tol: float = 0.005, maxiter: int = 20) -> dict:
    amg = dict(AMG_DEFAULTS)
    amg["smoother"] = cfg["coarse"]

    prec: dict = {
        "type":        cfg["ptype"],
        "weight_type": cfg["weight"],
        "pre_smooth":  0,
        "post_smooth": 1,
        "finesmoother": {
            "type":       cfg["fine"],
            "relaxation": 1.0,
        },
        "verbosity": 0,
        "coarsesolver": {
            "maxiter":       1,
            "tol":           0.1,
            "solver":        "loopsolver",
            "verbosity":     0,
            "preconditioner": amg,
        },
    }
    if cfg["ptype"] == "cprw":
        prec["use_well_weights"] = "true"
        prec["add_wells"]        = "true"

    return {
        "maxiter":       maxiter,
        "tol":           tol,
        "verbosity":     0,
        "solver":        "bicgstab",
        "preconditioner": prec,
    }


def write_configs(config_dir: Path) -> dict[int, Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for cfg in ALL_CONFIGS:
        p = config_dir / f"label_{cfg['label']:02d}.json"
        p.write_text(json.dumps(make_json(cfg), indent=2))
        paths[cfg["label"]] = p
    return paths


# ---------------------------------------------------------------------------
# Running flow
# ---------------------------------------------------------------------------

def run_one(flow: str, deck: str, cfg: dict, json_path: Path,
            out_dir: Path, csv_path: Path) -> tuple[int, bool, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "flow.log"
    env = os.environ.copy()
    env["OPM_CPR_SWEEP_LOG"] = str(csv_path)

    cmd = [
        flow,
        f"--linear-solver={json_path}",
        "--cpr-reuse-setup=0",
        f"--output-dir={out_dir}",
        deck,
    ]
    with open(log, "w") as fh:
        result = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return cfg["label"], result.returncode == 0, cfg["name"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow",    required=True, help="Path to flow executable")
    parser.add_argument("--decks",   nargs="+", required=True,
                        help="Reservoir deck files to sweep (.DATA)")
    parser.add_argument("--out",     default="/tmp/sweep2",
                        help="Output directory (default: /tmp/sweep2)")
    parser.add_argument("--jobs",    type=int, default=os.cpu_count() or 4,
                        help="Max parallel flow processes (default: nCPU)")
    parser.add_argument("--labels",  nargs="*", type=int,
                        help="Subset of labels 0-23 to run (default: all 24)")
    args = parser.parse_args()

    out_dir    = Path(args.out)
    config_dir = out_dir / "configs"

    configs = ALL_CONFIGS
    if args.labels:
        configs = [c for c in ALL_CONFIGS if c["label"] in args.labels]

    print(f"Writing {len(configs)} JSON configs → {config_dir}")
    json_paths = write_configs(config_dir)

    tasks = []
    for deck in args.decks:
        deck_stem = Path(deck).stem
        for cfg in configs:
            label     = cfg["label"]
            run_out   = out_dir / deck_stem / f"label_{label:02d}"
            csv_path  = out_dir / f"{deck_stem}_{label:02d}.csv"
            tasks.append((deck, cfg, json_paths[label], run_out, csv_path))

    print(f"Launching {len(tasks)} runs across {len(args.decks)} deck(s), "
          f"≤{args.jobs} in parallel ...")

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_one, args.flow, deck, cfg, jp, rod, csv): (deck, cfg)
            for deck, cfg, jp, rod, csv in tasks
        }
        for fut in as_completed(futures):
            deck, cfg = futures[fut]
            label, success, name = fut.result()
            deck_stem = Path(deck).stem
            status = "OK " if success else "FAIL"
            print(f"  [{status}] {deck_stem}  label={label:02d}  {name}")
            if success:
                ok += 1
            else:
                fail += 1

    print(f"\nDone: {ok} succeeded, {fail} failed.")
    if fail:
        print("Check flow.log files in the run subdirectories for errors.")

    csv_glob = str(out_dir / "*.csv")
    print(f"\nNext step:")
    print(f"  python collect_sweep_labels.py --csvs {csv_glob} --out training_data.csv")


if __name__ == "__main__":
    main()
