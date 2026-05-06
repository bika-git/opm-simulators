#!/usr/bin/env python3
"""
Combine per-Newton-step CSV files from the sweep and assign a label
(best config index) to each (deck, episode, newton_iter) sample.

Usage
-----
python collect_sweep_labels.py \\
    --csvs /tmp/sweep3/*.csv /tmp/sweep4/*.csv \\
    --out  training_data.csv

Each input CSV was produced by run_sweep.py / OPM_CPR_SWEEP_LOG.
The filename must follow the pattern  <deck_stem>_<NN>.csv  where NN is
the two-digit config label (0–23), as written by run_sweep.py.

One row per linear solve, columns:
  config, episode, newton_iter, total_ms, <feature cols...>

Output CSV has the same feature columns plus:
  deck          — reservoir model identifier (from filename)
  best_config   — winning config string ("cprw:trueimpes:dilu:ilu0" etc.)
  label         — integer 0-23 (joint 3×2×2×2 action index)
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config → (weight_type, use_cprw, fine_smoother, coarse_smoother)
# ---------------------------------------------------------------------------
# "type:weight:fine:coarse"   parsed from the logged config string

WEIGHT_TYPE = {"quasiimpes": 0, "trueimpes": 1, "trueimpesanalytic": 2}
CPRW        = {"cpr": 0, "cprw": 1}
FINE        = {"paroverilu0": 0, "dilu": 1}
COARSE      = {"ilu0": 0, "dilu": 1}


def config_to_label(config: str) -> int:
    parts = config.split(":")
    if len(parts) != 4:
        raise ValueError(f"Cannot parse config string: {config!r}")
    ptype, weight, fine, coarse = parts
    dec    = WEIGHT_TYPE.get(weight,  1)
    cprw   = CPRW.get(ptype,         1)
    f      = FINE.get(fine,           1)
    c      = COARSE.get(coarse,       0)
    return dec * 8 + cprw * 4 + f * 2 + c


FEATURE_COLS = [
    "dt_days", "dt_ratio", "nl_residual_norm", "nl_residual_reduce",
    "nl_iteration", "nnz_per_row", "diag_dominance",
    "prev_linsolver_iters", "prev_solve_failed",
    "time_elapsed_frac", "num_cells_log", "block_size",
    "well_density", "nl_residual_trend",         # added in sweep5
    "num_phases", "dt_cut_count", "prev2_linsolver_iters",
    "condition_number_estimate", "bhp_well_fraction",  # added in sweep6+
]

# Columns absent in older CSV files — fall back to "0".
OPTIONAL_FEATURE_COLS = {
    "well_density", "nl_residual_trend",
    "num_phases", "dt_cut_count", "prev2_linsolver_iters",
    "condition_number_estimate", "bhp_well_fraction",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csvs", nargs="+", required=True,
                        help="Sweep CSV files (one per config run)")
    parser.add_argument("--out",  default="training_data.csv",
                        help="Output CSV path (default: training_data.csv)")
    parser.add_argument("--min-configs", type=int, default=2,
                        help="Minimum configs per sample to keep (default: 2)")
    args = parser.parse_args()

    # Detect which feature columns are actually present across all input files.
    # Only emit columns that appear in at least one input CSV so that older
    # sweeps (fewer features) don't produce zero-padded trailing columns that
    # confuse the model.
    present_cols: set[str] = set()
    for path in args.csvs:
        with open(path, newline="") as fh:
            present_cols.update(csv.DictReader(fh).fieldnames or [])
    active_feature_cols = [c for c in FEATURE_COLS if c in present_cols]
    print(f"Feature columns detected: {len(active_feature_cols)} "
          f"({', '.join(active_feature_cols)})")

    # key → {config: [total_ms, ...], features: row_dict}
    samples: dict[tuple, dict] = defaultdict(lambda: {"times": {}, "feat": None})

    total_rows = 0
    for path in args.csvs:
        # Derive deck identifier from filename: strip the trailing _NN label suffix.
        stem = Path(path).stem          # e.g. "DROGON_HIST_23"
        deck = stem[:-3]               # strip "_23"  → "DROGON_HIST"

        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                total_rows += 1
                # Include deck so Newton steps from different reservoirs are
                # never merged together (they share episode/newton_iter numbering).
                key = (deck, row["episode"], row["newton_iter"])
                config = row["config"]
                ms = float(row["total_ms"])

                entry = samples[key]
                if config not in entry["times"]:
                    entry["times"][config] = []
                entry["times"][config].append(ms)

                if entry["feat"] is None:
                    entry["feat"] = {
                        c: row.get(c, "0") if c in OPTIONAL_FEATURE_COLS else row[c]
                        for c in active_feature_cols
                    }
                    entry["feat"]["deck"] = deck

    print(f"Read {total_rows} rows from {len(args.csvs)} files.")
    print(f"Unique (deck, episode, newton_iter) keys: {len(samples)}")

    out_rows = []
    skipped = 0
    for key, entry in samples.items():
        times = entry["times"]
        if len(times) < args.min_configs:
            skipped += 1
            continue

        # Use median time per config to reduce noise
        median_time = {cfg: sorted(ms)[len(ms) // 2]
                       for cfg, ms in times.items()}
        best_config = min(median_time, key=median_time.__getitem__)

        row = dict(entry["feat"])
        row["best_config"] = best_config
        row["label"]       = config_to_label(best_config)
        out_rows.append(row)

    print(f"Skipped {skipped} samples (fewer than {args.min_configs} configs).")
    print(f"Output samples: {len(out_rows)}")

    if not out_rows:
        print("ERROR: no output samples — check that multiple config CSVs cover the "
              "same (episode, newton_iter) keys.", file=sys.stderr)
        sys.exit(1)

    # Label distribution
    from collections import Counter
    dist = Counter(r["label"] for r in out_rows)
    print("\nLabel distribution (top 10):")
    for label, count in dist.most_common(10):
        print(f"  label {label:2d}: {count:5d} samples")

    fieldnames = ["deck"] + active_feature_cols + ["best_config", "label"]
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} samples → {args.out}")


if __name__ == "__main__":
    main()
