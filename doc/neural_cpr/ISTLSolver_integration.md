# Module: ISTLSolver Integration

**File:** `opm/simulators/linalg/ISTLSolver.hpp`

---

## Overview

The neural policies are wired into `ISTLSolver<TypeTag>` with minimal
surgical changes. No new virtual methods, no new class hierarchy, no changes
outside this file. All policies are additive: when their environment variables
are unset the solver behaves identically to the pre-patch baseline.

Three policies integrate here:

| Policy | Integration point | Env vars |
|---|---|---|
| CPR variant | `applyNeuralPolicy()` in `prepare()` | `OPM_NEURAL_CPR_WEIGHTS` |
| Neural AMG | `applyNeuralPolicy()` in `prepare()` | `OPM_NEURAL_CPR_WEIGHTS` + `OPM_NEURAL_CPR_ACTIONS=amg` |
| Neural abort | `applyAbortPolicy()` in `solve()` | `OPM_CPR_ABORT_MODEL` |

---

## Hook locations

### CPR variant and AMG — `prepare()`

```
prepare(M, b)
  └── initPrepare(M, b)       ← existing: stores matrix pointer, checks overlap rows
  └── applyNeuralPolicy(M)    ← NEW: evaluate CPR variant or AMG policy, patch prm_
  └── prepareFlexibleSolver() ← existing: builds/updates solver from prm_
```

`applyNeuralPolicy` runs between matrix inspection and solver construction,
so `rhs_` and `matrix_` are valid and `prm_[activeSolverNum_]` is overwritten
before `FlexibleSolver::create()` reads it.

### Neural abort — `solve()`

```
solve(M, x, b)
  └── (existing solve loop with Newton iterations)
  └── applyAbortPolicy()      ← NEW: check abort model after each linearisation
      └── if prob > threshold: signal early exit to NonlinearSolver
```

The abort policy runs inside the solve loop, not in `prepare()`, because it
needs the per-linearisation residual trajectory that only exists during
solving.

---

## New methods

### `initNeuralPolicy()`

Called once at the end of `initialize()`. Reads `OPM_NEURAL_CPR_WEIGHTS` and
`OPM_NEURAL_CPR_ACTIONS`; constructs the appropriate policy object. Reads
`OPM_CPR_ABORT_MODEL` and initialises the abort policy. Logs to
`OpmLog::info` on rank 0.

### `extractFeatures(const Matrix& M) → NeuralCPRFeatures`

Assembles up to 19 feature values from live simulator state. The first 12
core features are always populated; extended features are populated where the
simulator exposes them:

```cpp
simulator_.timeStepSize()                            // dt_days, dt_ratio
rhs_->two_norm()                                     // nl_residual_norm
simulator_.problem().iterationContext().iteration()  // nl_iteration
M.nonzeroes() / M.N()                               // nnz_per_row  (cached)
computeDiagDominance(M)                              // diag_dominance (cached)
// … prev_linsolver_iters, prev_solve_failed, time_elapsed_frac,
//     num_cells_log, block_size, well_density, nl_residual_trend,
//     num_phases, dt_cut_count, prev2_linsolver_iters,
//     condition_number_estimate, bhp_well_fraction
```

`prev_dt_` and `prev_residual_norm_` are stored across calls to compute the
ratio features.

### `computeDiagDominance(const Matrix& M) → double` (static)

Samples up to 200 block rows and computes:

```
mean over sampled rows of: ||a_ii||_F / Σ_{j≠i} ||a_ij||_F
```

Cost: O(200 × nnz_per_row × numEq²) — negligible relative to AMG setup.

### `applyNeuralPolicy(const Matrix& M)`

Implements the three-tier caching strategy (see below) and writes the chosen
configuration into `prm_[activeSolverNum_]`. When `OPM_NEURAL_CPR_ACTIONS=amg`
is set, writes the AMG damping and coarsening target instead of the CPR
variant config.

### `applyAbortPolicy()`

Reads the abort model's probability output. If it exceeds
`OPM_CPR_ABORT_THRESHOLD` (default 0.95), sets a flag that causes the outer
`NonlinearSolver` loop to treat the current attempt as a chop.

### `policyLogStr(const CPRConfig&) → std::string` (static)

Returns a one-line human-readable description of a config for `OpmLog::debug`.

---

## Three-tier caching (CPR variant and AMG)

The policy runs on the critical path of every Newton iteration. Three
successive guards ensure the common case costs essentially nothing.

### Tier 1 — `shouldCreateSolver()` early exit

```cpp
if (!shouldCreateSolver())
    return;
```

`prm_[activeSolverNum_]` is only consumed when `shouldCreateSolver()` returns
`true`. In the default reuse mode (`--cpr-reuse-setup=4`), this returns `true`
only every 4 Newton iterations. For a typical 2–3 Newton iteration timestep
this skips the policy on most calls at zero cost.

### Tier 2 — config equality check

```cpp
if (policy_cfg_valid_ && cfg == last_policy_cfg_)
    return;
```

Even when a rebuild is due, the predicted config often matches the previous
one. `CPRConfig::operator==` compares four enum/bool fields, O(1).

### Tier 3 — selective field update

```cpp
if (policy_cfg_valid_ && cfg.use_cprw == last_policy_cfg_.use_cprw)
    NeuralCPRPolicy::updatePropertyTreeInPlace(cfg, last_policy_cfg_, prm_);
else
    prm_[activeSolverNum_] = NeuralCPRPolicy::configToPropertyTree(cfg, tol, maxiter);
```

When the preconditioner type (`cpr` vs `cprw`) is unchanged, only the 1–3
fields that differ are updated via individual `put()` calls, avoiding a full
30-node Boost property tree rebuild. When the type changes, the existing
`FlexibleSolver` is discarded so it is rebuilt with the new tree.

---

## New member variables

```cpp
// Policy engines
std::unique_ptr<NeuralCPRPolicy>   neural_policy_;    // CPR variant or AMG
std::unique_ptr<NeuralAbortPolicy> abort_policy_;     // early-abort

// Convergence tracking
bool   last_solve_failed_  = false;

// Tier-2 caching
bool      policy_cfg_valid_ = false;
CPRConfig last_policy_cfg_;

// Feature trajectory tracking
double prev_dt_            = 0.0;
double prev_residual_norm_ = 0.0;
int    prev_linsolver_iters_ = 0;
bool   prev_solve_failed_stat_ = false;

// Invariant feature caches
mutable double cached_nnz_per_row_    = -1.0;  // permanent after first call
mutable double cached_diag_dominance_ =  1.0;  // refreshed on Newton iter 0
mutable double cached_num_cells_log_  = -1.0;  // permanent
mutable double cached_block_size_     = -1.0;  // permanent
```

---

## Convergence-failure fallback

`solve()` records whether the linear solve converged:

```cpp
last_solve_failed_ = !checkConvergence(result);
```

`applyNeuralPolicy()` checks this flag and falls back to the safe default
config (trueimpes + dilu) when the previous solve failed. This prevents the
policy from repeating a bad choice on the retry after a timestep chop.

---

## Impact on existing behaviour

| Scenario | Impact |
|---|---|
| No neural env vars set | Rule-based CPR variant active; zero overhead in default reuse mode |
| `OPM_NEURAL_CPR_WEIGHTS` set, file valid | MLP inference for CPR variant; same caching guarantees |
| `OPM_NEURAL_CPR_WEIGHTS` + `ACTIONS=amg` | AMG parameter policy active instead of CPR variant |
| `OPM_NEURAL_CPR_DISABLE=1` | CPR variant disabled; abort policy can still run alone |
| `OPM_CPR_ABORT_MODEL` set | Abort policy runs inside `solve()` after each linearisation |
| File path invalid | Silent fallback to rule-based; no crash |
| `--cpr-reuse-setup=0` | Tier 1 never exits early; policy runs every Newton iteration |
| `--cpr-reuse-setup=4` (default) | Tier 1 skips most calls; maximum reuse |
| Parallel (MPI) | Policy runs independently on all ranks; `OpmLog` messages on rank 0 only |

---

## CMakeLists_files.cmake

New headers registered in `PUBLIC_HEADER_FILES`:

```cmake
opm/simulators/linalg/NeuralCprFeatures.hpp
opm/simulators/linalg/NeuralCprPolicy.hpp
opm/simulators/linalg/NeuralAbortPolicy.hpp
```

No new `.cpp` files or `find_package` calls needed beyond the existing
Boost ptree dependency.
