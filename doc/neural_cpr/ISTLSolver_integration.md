# Module: ISTLSolver Integration

**File:** `opm/simulators/linalg/ISTLSolver.hpp`

---

## Overview

The neural policy is wired into the existing `ISTLSolver<TypeTag>` class
template with minimal surgical changes.  No new virtual methods, no new
class hierarchy, no changes outside this file.  The policy is entirely
additive: when the environment variable `OPM_NEURAL_CPR_WEIGHTS` is unset
and no weights file is found, the solver behaves identically to the
pre-patch baseline.

---

## Hook location

The sole integration point is `ISTLSolver::prepare(const Matrix&, Vector&)`:

```
prepare(M, b)
  └── initPrepare(M, b)       ← existing: stores matrix pointer, checks overlap rows
  └── applyNeuralPolicy(M)    ← NEW: evaluate policy, patch prm_ if needed
  └── prepareFlexibleSolver() ← existing: builds/updates solver from prm_
```

`applyNeuralPolicy` runs between the matrix inspection phase and the solver
construction phase.  This ordering guarantees:
- `rhs_` and `matrix_` are valid (set by `initPrepare`)
- `prm_[activeSolverNum_]` is overwritten before `FlexibleSolver::create()`
  reads it
- The existing `shouldCreateSolver()` logic (reuse strategy, AMG rebuild
  intervals) is unmodified and still controls whether the solver is actually
  rebuilt

---

## New methods

### `initNeuralPolicy()`

Called once at the end of `initialize()`.  Reads the environment variable
`OPM_NEURAL_CPR_WEIGHTS`; constructs a `NeuralCPRPolicy` from the path if
set, otherwise constructs one with an empty path (rule-based fallback).
Logs to `OpmLog::info` on rank 0.

### `extractFeatures(const Matrix& M) → NeuralCPRFeatures`

Assembles the eight feature values from live simulator state:

```cpp
simulator_.timeStepSize()                           // dt_days, dt_ratio
rhs_->two_norm()                                    // nl_residual_norm
simulator_.problem().iterationContext().iteration() // nl_iteration
M.nonzeroes() / M.N()                              // nnz_per_row  (cached)
computeDiagDominance(M)                             // diag_dominance (cached)
```

`prev_dt_` and `prev_residual_norm_` are stored across calls to compute the
ratio features.

### `computeDiagDominance(const Matrix& M) → double` (static)

Samples up to 200 block rows of the matrix and computes the mean ratio:

```
mean over sampled rows of: ||a_ii||_F / Σ_{j≠i} ||a_ij||_F
```

`frobenius_norm()` is already defined on OPM's `MatrixBlock`/`FieldMatrix`.
Cost: O(200 × nnz_per_row × numEq²) — negligible relative to AMG setup.

### `applyNeuralPolicy(const Matrix& M)`

Implements the three-tier caching strategy (see below) and writes the chosen
configuration into `prm_[activeSolverNum_]`.

### `policyLogStr(const CPRConfig&) → std::string` (static)

Returns a one-line human-readable description of a config for `OpmLog::debug`.

---

## Three-tier caching

The policy is on the critical path of every Newton iteration.  Three
successive guards ensure the common case (stable config, solver not being
rebuilt) costs essentially nothing.

### Tier 1 — `shouldCreateSolver()` early exit

```cpp
if (!shouldCreateSolver())
    return;
```

`prm_[activeSolverNum_]` is only consumed by `FlexibleSolver::create()`,
which is called only when `shouldCreateSolver()` returns `true`.  In the
default reuse mode (`--cpr-reuse-setup=1`), this returns `true` only on the
first Newton iteration of each timestep.  For a typical case with 2–3 Newton
iterations per timestep, this skips the policy on 50–67% of `prepare()`
calls at zero cost.

### Tier 2 — config equality check

```cpp
if (policy_cfg_valid_ && cfg == last_policy_cfg_)
    return;
```

Even when the solver is being rebuilt, the predicted config often matches the
previous one (the simulation is in a stable phase).  This guard avoids all
`PropertyTree` manipulation when nothing has changed.  `CPRConfig::operator==`
compares four enum/bool fields, O(1).

### Tier 3 — selective field update

```cpp
if (policy_cfg_valid_ && cfg.use_cprw == last_policy_cfg_.use_cprw)
    NeuralCPRPolicy::updatePropertyTreeInPlace(cfg, last_policy_cfg_, prm_);
else
    prm_[activeSolverNum_] = NeuralCPRPolicy::configToPropertyTree(cfg, tol, maxiter);
```

When the preconditioner type (`cpr` vs `cprw`) is unchanged, only the 1–3
fields that actually differ are updated via individual `put()` calls.  This
avoids the full 30-node Boost property tree rebuild.

When the type changes (rare in practice), the existing `FlexibleSolver`
object must be discarded since it was built for a different preconditioner:

```cpp
if (policy_cfg_valid_ && flexibleSolver_[activeSolverNum_].solver_)
    flexibleSolver_[activeSolverNum_].solver_.reset();
```

Nulling the solver pointer causes `shouldCreateSolver()` to return `true`
on the next call, which then calls `FlexibleSolver::create()` with the new
tree.

---

## New member variables

```cpp
// Policy engine
std::unique_ptr<NeuralCPRPolicy> neural_policy_;

// Convergence tracking
bool   last_solve_failed_  = false;  // set in solve(); gates safe-default fallback

// Tier-2 caching
bool      policy_cfg_valid_ = false; // true once last_policy_cfg_ is set
CPRConfig last_policy_cfg_;          // config last written into prm_

// Features: trajectory tracking (updated every applyNeuralPolicy call)
double prev_dt_            = 0.0;
double prev_residual_norm_ = 0.0;

// Features: invariant caches
mutable double cached_nnz_per_row_    = -1.0; // permanent after first call
mutable double cached_diag_dominance_ =  1.0; // refreshed on Newton iter 0
```

All members have in-class initialisers so they work with both ISTLSolver
constructors without modification.

---

## Convergence-failure fallback

`solve()` now records whether the linear solve converged:

```cpp
last_solve_failed_ = !checkConvergence(result);
return !last_solve_failed_;
```

`applyNeuralPolicy()` checks this flag:

```cpp
if (last_solve_failed_)
    cfg = CPRConfig{};   // default-constructed = safe trueimpes+DILU config
```

This guarantees that if the policy emits a configuration that causes the
linear solver to fail (diverge or exceed `maxiter`), the next solve attempt
uses the known-good default.  The timestep-chopping logic in `NonlinearSolver`
provides the outer retry loop; the flag ensures the policy does not repeat a
bad choice on the retry.

---

## Impact on existing behaviour

| Scenario | Impact |
|---|---|
| `OPM_NEURAL_CPR_WEIGHTS` not set | Rule-based policy active; zero overhead in default reuse mode |
| `OPM_NEURAL_CPR_WEIGHTS` set, file valid | MLP inference; same caching guarantees |
| `OPM_NEURAL_CPR_WEIGHTS` set, file invalid | Silently falls back to rule-based |
| `--cpr-reuse-setup=3` (never rebuild) | Tier 1 exits immediately every call; policy never runs |
| `--cpr-reuse-setup=0` (always rebuild) | Policy runs every Newton iteration; maximum adaptation |
| `--linear-solver` not set to `cpr*` | Policy still runs but writes `cprw` tree; `shouldCreateSolver` controls rebuild |
| Parallel (MPI) runs | Policy runs on all ranks independently; `OpmLog` messages emitted on rank 0 only |

---

## CMakeLists_files.cmake

The two new headers are added to `PUBLIC_HEADER_FILES`:

```cmake
opm/simulators/linalg/NeuralCPRFeatures.hpp
opm/simulators/linalg/NeuralCPRPolicy.hpp
```

No new `.cpp` files or `find_package` calls are needed.  `PropertyTree.hpp`
and its Boost ptree dependency are already transitively available from the
existing `setupPropertyTree.hpp` include chain.
