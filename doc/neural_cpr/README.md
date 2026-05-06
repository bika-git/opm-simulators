# Neural CPR Policy

A learned linear-solver configuration layer that replaces the static
`--linear-solver=cprw` setting with a per-Newton-iteration decision,
choosing the CPR preconditioner parameters that minimise total linear-solve
time based on the current state of the simulator.

---

## Motivation

OPM Flow's CPR (Constrained Pressure Residual) preconditioner exposes three
independent axes of configuration:

| Axis | Options | Effect |
|---|---|---|
| Pressure-decoupling | `quasiimpes`, `trueimpes`, `trueimpesanalytic` | Accuracy vs. cost of the pressure extraction weights |
| Fine smoother | `paroverilu0`, `dilu` | ILU vs. decoupled-ILU before the coarse correction |
| Coarse AMG smoother | `ilu0`, `dilu` | Smoother inside the AMG V-cycle |

The standard approach picks one combination at startup and keeps it for the
entire simulation.  But the difficulty of each linear solve changes
continuously: the first Newton step of a timestep is structurally easy (the
system is close to the previous solution), while later steps or rapid
well-schedule changes can produce ill-conditioned systems that need the
more accurate (and more expensive) `trueimpes` decoupling.

The neural CPR policy adapts these three choices every time the solver is
rebuilt, using eight lightweight features extracted from the current matrix
and Newton state to predict which configuration minimises wall time.

---

## Architecture

```
simulator state  ──┐
                   ├──► NeuralCPRFeatures ──► NeuralCPRPolicy ──► CPRConfig
assembled matrix ──┘          (8 features)        (MLP / rules)

CPRConfig ──► NeuralCPRPolicy::configToPropertyTree()
          ──► prm_[activeSolverNum_]          (in-place update)
          ──► FlexibleSolver::create()        (existing OPM path)
```

The hook lives inside `ISTLSolver::prepare()`, between the existing
`initPrepare()` and `prepareFlexibleSolver()` calls.  No changes to any
other part of the simulation pipeline are required.

---

## Files changed / added

| File | Role |
|---|---|
| `opm/simulators/linalg/NeuralCprFeatures.hpp` | Feature struct + normalisation |
| `opm/simulators/linalg/NeuralCprPolicy.hpp` | MLP, rule-based fallback, PropertyTree builder |
| `opm/simulators/linalg/ISTLSolver.hpp` | Integration hook + caching |
| `CMakeLists_files.cmake` | Header registration |

Detailed documentation for each module:

- [NeuralCPRFeatures](NeuralCPRFeatures.md)
- [NeuralCPRPolicy](NeuralCPRPolicy.md)
- [ISTLSolver integration](ISTLSolver_integration.md)

---

## Quick start

### Default mode (rule-based fallback, zero overhead)

No arguments needed.  The policy activates automatically and uses the
rule-based heuristic when no weights file is present:

```bash
flow --linear-solver=cprw DECK.DATA
```

The `.DBG` log will show one line per solver rebuild:

```
NeuralCPRPolicy: active (rule-based fallback)
NeuralCPRPolicy: cprw weight=quasiimpes fine=paroverilu0 coarse=ilu0
NeuralCPRPolicy: cprw weight=trueimpes fine=dilu coarse=ilu0
```

### With trained MLP weights

```bash
export OPM_NEURAL_CPR_WEIGHTS=/path/to/weights.bin
flow --linear-solver=cprw DECK.DATA
```

The binary weight file format is documented in [NeuralCPRPolicy](NeuralCPRPolicy.md#weight-file-format).

### Experimental: rebuild AMG on every Newton step

Forces the policy to be evaluated at every Newton iteration instead of
only on the first one per timestep.  Reduces linear iterations but
increases setup overhead; useful for training-data collection:

```bash
OPM_NEURAL_CPR_WEIGHTS=/path/to/weights.bin \
  flow --linear-solver=cprw --cpr-reuse-setup=0 DECK.DATA
```

---

## Benchmark results

Tested with the `bikagit/opm-simulators` build against OPM test cases.

### SPE9 (9,000 cells, 329 linearizations)

| Configuration | Sim time | Linear setup | Linear iters |
|---|---|---|---|
| Baseline `cprw` (static trueimpes) | 6.55 s | 1.34 s | 481 |
| Neural policy, default reuse | 6.56 s | 1.34 s | 481 |
| Neural policy, `--cpr-reuse-setup=0` | 9.22 s | 4.19 s | **418 (−13%)** |

### Norne (45,217 cells, 1,333 Newton iterations)

| Configuration | Sim time | Linear setup | Linear iters |
|---|---|---|---|
| Baseline `cprw` (static trueimpes) | 185.1 s | 26.6 s | 2452 |
| Neural policy, default reuse | 184.5 s | 26.8 s | 2452 |

**Key findings:**

- In default reuse mode the neural policy adds **zero measurable overhead**
  (three-tier caching eliminates all redundant work when the config is stable).
- With `--cpr-reuse-setup=0` the rule-based policy already reduces linear
  iterations by 13% on SPE9 by selecting `quasiimpes` on the first Newton
  step of each timestep, where the cheaper weight computation does not
  compromise solver quality.
- The full benefit of trained MLP weights will emerge on larger, more
  heterogeneous cases where the optimal config varies significantly across
  timesteps.

---

## Training pipeline (Python)

See the companion training script outline in [NeuralCPRPolicy](NeuralCPRPolicy.md#training-pipeline).
The data-collection sweep uses the existing `--linear-solver=cpr_trueimpes`,
`--linear-solver=cpr_quasiimpes`, and `--linear-solver=cprw` flags to evaluate
all config combinations on benchmark decks.  Labels are wall-time-per-linear-solve.
