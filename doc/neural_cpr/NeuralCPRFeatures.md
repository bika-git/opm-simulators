# Module: NeuralCPRFeatures

**File:** `opm/simulators/linalg/NeuralCPRFeatures.hpp`

---

## Purpose

`NeuralCPRFeatures` is a plain data struct that captures eight scalar
quantities describing the current state of the nonlinear solver at the
moment `ISTLSolver::prepare()` is called.  Its only method,
`toInputVector()`, normalises these quantities to a uniform `[0, 1]` range
suitable for MLP inference.

The struct deliberately has no dependencies beyond the standard library so
it can be included anywhere in the solver stack without pulling in OPM
headers.

---

## The eight features

| Index | Field | Unit | Description |
|---|---|---|---|
| 0 | `dt_days` | days | Current timestep length |
| 1 | `dt_ratio` | — | `dt / dt_prev`; 1.0 on the first step |
| 2 | `nl_residual_norm` | — | L2 norm of the assembled right-hand side |
| 3 | `nl_residual_reduce` | — | `norm / norm_at_Newton_iter_0` for this timestep |
| 4 | `nl_iteration` | — | Newton iteration index (0-based) |
| 5 | `nnz_per_row` | — | Average non-zeros per block row |
| 6 | `diag_dominance` | — | Mean `‖a_ii‖_F / Σ_{j≠i} ‖a_ij‖_F` over 200 sampled rows |
| 7 | `pad` | — | Reserved, always 0 |

### Feature rationale

**`dt_days` (log-scaled):** Large timesteps produce stiffer Jacobians that
require more robust (expensive) decoupling strategies.  The logarithmic
scale compresses the dynamic range; reservoir timesteps span four orders of
magnitude.

**`dt_ratio`:** A sudden increase in stepsize (ratio ≫ 1) is a reliable
predictor of a harder solve.  A ratio < 1 (timestep cut after a failed step)
signals that the previous configuration was insufficient.

**`nl_residual_norm`:** The raw norm distinguishes initial linearisations
(high norm, problem easy) from later Newton steps (low norm, near-convergence,
problem may be ill-conditioned).

**`nl_residual_reduce`:** The reduction ratio tracks Newton convergence rate
within a timestep.  A ratio close to 1 (residual barely decreasing) is the
strongest single predictor that a more robust solver configuration is needed.

**`nl_iteration`:** Directly encodes how many Newton steps have already been
taken.  Early steps tolerate cheaper configs; repeated attempts indicate difficulty.

**`nnz_per_row`:** Captures problem connectivity.  Highly connected systems
(e.g., corner-point grids with many active neighbours) form denser Jacobians
where AMG smoother choice matters more.  Constant throughout a simulation;
cached after the first call.

**`diag_dominance`:** The ratio of diagonal-block Frobenius norm to the sum
of off-diagonal norms indicates how well the pressure subsystem is separated
from the saturation equations.  Low dominance → `trueimpes` decoupling helps
more.  Recomputed only on Newton iteration 0 of each timestep.

**`pad`:** Placeholder for a future feature (e.g., well density) without
changing the MLP input dimension.

---

## Normalisation (`toInputVector`)

All eight slots are mapped to roughly `[0, 1]` before feeding the MLP:

```
x[0] = log1p(dt_days) / 7.0
x[1] = clamp(dt_ratio / 10.0,  0, 1)
x[2] = (log10(max(norm, 1e-14)) + 14.0) / 14.0   # maps [-14,0] → [0,1]
x[3] = clamp(nl_residual_reduce, 0, 1)
x[4] = clamp(nl_iteration / 20.0, 0, 1)
x[5] = log1p(nnz_per_row) / 5.0
x[6] = clamp(diag_dominance / 4.0, 0, 1)
x[7] = 0.0
```

The log transforms handle the wide dynamic ranges typical in reservoir
simulation without requiring per-run standardisation.

---

## Caching behaviour (inside ISTLSolver)

Two features are cached across calls to avoid redundant computation:

| Feature | Cache policy | Justification |
|---|---|---|
| `nnz_per_row` | Permanent — computed once | Sparsity pattern never changes |
| `diag_dominance` | Per-timestep — recomputed when `nl_iteration == 0` | Computation is O(200 rows); values change slowly across Newton steps |

`nl_residual_norm`, `nl_residual_reduce`, and `nl_iteration` are cheap to
compute (L2 norm of the rhs vector + two lookups) and always reflect the
current Newton state, so they are never cached.

---

## Adding a new feature

1. Add a new field to `NeuralCPRFeatures` with a sensible default.
2. Update `toInputVector()` — shift the new value into an existing slot or
   replace `pad`.  Keep the output dimension at 8 to avoid changing the MLP
   weight shapes.
3. Update `ISTLSolver::extractFeatures()` to populate the field.
4. Retrain the MLP with data collected using the updated features.
