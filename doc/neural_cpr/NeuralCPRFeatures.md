# Module: NeuralCPRFeatures

**File:** `opm/simulators/linalg/NeuralCprFeatures.hpp`

---

## Purpose

`NeuralCPRFeatures` is a plain data struct that captures scalar quantities
describing the current state of the nonlinear solver at the moment
`ISTLSolver::prepare()` is called. Its `toInputVector()` method normalises
these to a `[0, 1]` range suitable for MLP inference.

---

## CPR variant policy features

The CPR variant policy (`train_cpr_policy.py`) uses up to 19 features. The
first 12 are the **core** columns always present in sweep CSVs; the remaining
7 are **extended** columns added later (older CSVs default them to 0).

### Core features (indices 0–11)

| Index | Field | Normalisation | Description |
|---|---|---|---|
| 0 | `dt_days` | `log1p(x) / 7` | Current timestep length in days |
| 1 | `dt_ratio` | `clip(x / 10, 0, 1)` | `dt / dt_prev`; 1.0 on the first step |
| 2 | `nl_residual_norm` | `(log10(max(x,1e-14)) + 14) / 14` | L2 norm of the assembled right-hand side |
| 3 | `nl_residual_reduce` | `clip(x, 0, 1)` | `norm / norm_at_Newton_iter_0` |
| 4 | `nl_iteration` | `clip(x / 20, 0, 1)` | Newton iteration index (0-based) |
| 5 | `nnz_per_row` | `log1p(x) / 5` | Average non-zeros per block row |
| 6 | `diag_dominance` | `clip(x / 4, 0, 1)` | Mean `‖a_ii‖_F / Σ_{j≠i} ‖a_ij‖_F` over 200 sampled rows |
| 7 | `prev_linsolver_iters` | `clip(x / 50, 0, 1)` | Linear iterations in the previous Newton step |
| 8 | `prev_solve_failed` | `clip(x, 0, 1)` | 1 if the previous linear solve failed to converge |
| 9 | `time_elapsed_frac` | `clip(x, 0, 1)` | Fraction of total simulation time elapsed |
| 10 | `num_cells_log` | `clip(x / 6, 0, 1)` | `log10(num_cells)` (e.g. 5 → 100k cells) |
| 11 | `block_size` | `clip(x / 6, 0, 1)` | Number of equations per cell (3 = oil/gas/water) |

### Extended features (indices 12–18, default 0 for old CSVs)

| Index | Field | Normalisation | Description |
|---|---|---|---|
| 12 | `well_density` | `clip(x * 200, 0, 1)` | Fraction of cells that are well connections |
| 13 | `nl_residual_trend` | `clip((log10(max(x,0.01)) + 2) / 4, 0, 1)` | Ratio of current to previous residual norm (convergence rate) |
| 14 | `num_phases` | `clip(x / 3, 0, 1)` | Number of active fluid phases |
| 15 | `dt_cut_count` | `clip(x / 5, 0, 1)` | Number of consecutive timestep cuts |
| 16 | `prev2_linsolver_iters` | `clip(x / 50, 0, 1)` | Linear iterations two Newton steps ago |
| 17 | `condition_number_estimate` | `clip(log10(max(x,1)) / 10, 0, 1)` | Estimated matrix condition number |
| 18 | `bhp_well_fraction` | `clip(x, 0, 1)` | Fraction of wells on BHP control |

### Feature rationale (selected)

**`dt_days` (log-scaled):** Large timesteps produce stiffer Jacobians. The
log scale compresses a dynamic range spanning four orders of magnitude.

**`dt_ratio`:** A sudden increase (ratio ≫ 1) predicts a harder solve. A
ratio < 1 signals a recent timestep cut.

**`nl_residual_reduce`:** Ratio close to 1 (residual barely decreasing) is
the strongest single predictor that a more robust solver config is needed.

**`prev_linsolver_iters`:** High iteration count on the previous step signals
that the system is approaching the solver's convergence limit.

**`diag_dominance`:** Low dominance → `trueimpes` decoupling helps more.
Recomputed only on Newton iteration 0 of each timestep.

**`nnz_per_row`:** Captures grid connectivity; constant throughout a
simulation, cached after the first call.

---

## AMG policy features

The AMG policy (`train_cpr_amg_policy.py`) uses the first 12 core features
above (`N_FEAT = 12`), evaluated at CPR setup rebuild points.

---

## Abort policy features

The abort policy (`train_abort_policy.py`) uses 14 features extracted from
the Newton convergence report and the CPR solver's own state. These are what
the linear solver itself can observe at each Newton step:

| Group | Features |
|---|---|
| `SOLVER_KNOWN` (5) | `iteration`, `after_chop`, `chop_streak`, `dt_change`, `prev_iters` |
| `CPR_FEATURES` (4) | `last_lin_its`, `max_lin_its`, `setup_age`, `rebuilt_now` |
| `RES_FEATURES` (5) | `log_res`, `res_vs_start`, `res_step`, `res_vs_best`, `res_frac_up` |

Input standardisation `(x − μ) / σ` is folded into the first weight matrix
at export time, so Flow feeds raw (un-normalised) values.

---

## Caching behaviour (inside ISTLSolver)

| Feature | Cache policy | Justification |
|---|---|---|
| `nnz_per_row` | Permanent — computed once | Sparsity pattern never changes |
| `diag_dominance` | Per-timestep — refreshed when `nl_iteration == 0` | O(200 rows); changes slowly across Newton steps |
| `num_cells_log`, `block_size` | Permanent — from grid metadata | Grid and physics are fixed |
| All others | Live — read each call | Cheap or must reflect current state |
