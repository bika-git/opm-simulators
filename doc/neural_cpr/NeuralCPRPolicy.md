# Module: NeuralCPRPolicy

**File:** `opm/simulators/linalg/NeuralCprPolicy.hpp`

---

## Purpose

`NeuralCPRPolicy` is the decision engine.  Given a `NeuralCPRFeatures`
struct it produces a `CPRConfig` — a discrete choice across three solver
axes — and can serialise that choice into an `Opm::PropertyTree` that
`FlexibleSolver::create()` consumes directly.

Two operating modes:

| Mode | Condition | Behaviour |
|---|---|---|
| **MLP inference** | Weight file loaded (`isLoaded() == true`) | Forward pass through 8→32→16→8 network |
| **Rule-based fallback** | No weight file / file unreadable | Deterministic heuristic, no file I/O |

---

## CPRConfig

```cpp
struct CPRConfig {
    CPRDecoupling    decoupling;   // QuasiIMPES | TrueIMPES | TrueIMPESAnalytic
    bool             use_cprw;    // true → "cprw", false → "cpr"
    CPRFineSmoother  fine;        // ParOverILU0 | DILU
    CPRCoarseSmoother coarse;     // ILU0 | DILU
};
```

`CPRConfig` has value semantics and provides `operator==` / `operator!=`,
which the caching layer in `ISTLSolver` uses to detect when the config has
changed and a PropertyTree update is needed.

---

## MLP architecture

```
input   (8)  ──► Linear(8→32) + ReLU
hidden  (32) ──► Linear(32→16) + ReLU
hidden  (16) ──► Linear(16→8)
logits  (8)  ──► decode
```

Output decoding (argmax / threshold per group):

| Logit slice | Decision |
|---|---|
| `[0:3]` — 3 values | `argmax` → decoupling (0=QuasiIMPES, 1=TrueIMPES, 2=Analytic) |
| `[3]` — scalar | `> 0` → `use_cprw = true` |
| `[4:6]` — 2 values | `argmax` → fine smoother (0=ParOverILU0, 1=DILU) |
| `[6:8]` — 2 values | `argmax` → coarse smoother (0=ILU0, 1=DILU) |

Total output dimension: 8 logits encoding 3 + 1 + 2 + 2 independent decisions.
Each decision head is trained with separate cross-entropy loss against the
label produced by the data-collection sweep.

---

## Rule-based fallback

When no weight file is available the policy uses a deterministic heuristic
that captures the most important insight from the OPM documentation: the
first Newton step of each timestep is structurally easier and tolerates
cheaper weight computation.

```
Newton iter 0 (first linearisation of a new timestep):
    → quasiimpes + ParOverILU0 + ILU0
    Justification: quasiimpes computes weights from storage terms only
    (O(n) diagonal operations vs. O(n × numEq³) local solves for trueimpes).
    ParOverILU0 setup is also faster than DILU for well-conditioned systems.

Slow convergence (iter > 0 and residual_reduce > 0.5) or many iterations (≥ 4):
    → trueimpes + DILU + ILU0
    Justification: the more accurate decoupling and more robust smoother
    are needed when the Newton sequence is struggling.

Otherwise (iter 1–3, converging well):
    → quasiimpes + DILU + ILU0
    Balanced: cheap decoupling, robust smoother.
```

Benchmark impact of the rule-based fallback on SPE9 with `--cpr-reuse-setup=0`:
**−13% linear iterations** (418 vs 481) compared to the static `trueimpes`
baseline, because the cheaper quasiimpes config on Newton iter 0 produces a
different AMG hierarchy that converges in fewer Krylov steps on that particular
call.

---

## PropertyTree construction

`configToPropertyTree(cfg, tol, maxiter)` produces a complete Boost property
tree that mirrors the output of `setupCPRW()` in `setupPropertyTree.cpp` line
for line.  All 20+ fields are populated with the same defaults as the
standard `cprw` configuration; only the three axis-specific fields differ:

| Property tree key | Controlled by |
|---|---|
| `preconditioner.type` | `cfg.use_cprw` → `"cprw"` or `"cpr"` |
| `preconditioner.weight_type` | `cfg.decoupling` → `"quasiimpes"` / `"trueimpes"` / `"trueimpesanalytic"` |
| `preconditioner.finesmoother.type` | `cfg.fine` → `"dilu"` or `"paroverilu0"` |
| `preconditioner.coarsesolver.preconditioner.smoother` | `cfg.coarse` → `"ilu0"` or `"dilu"` |

All other AMG parameters (`alpha`, `coarsenTarget`, `maxaggsize`, etc.) are
identical to the OPM defaults from `setupDuneAMG()`.

`updatePropertyTreeInPlace(cfg, prev, prm)` is the selective-update variant:
when the preconditioner type is unchanged it calls `put()` only for the 1–3
fields that differ, avoiding a full 30-node Boost ptree rebuild.

---

## Weight file format

The binary weight file is a compact flat layout, with no external library
dependency for reading:

```
Offset   Size    Content
0        4 B     Magic: 0x4E435052 ('NCPR' little-endian uint32)
4        4 B     Version: 1 (uint32)
8        1024 B  W1[32][8]  — layer-1 weight matrix, row-major, float32
1032     128 B   b1[32]     — layer-1 bias vector, float32
1160     2048 B  W2[16][32] — layer-2 weight matrix, row-major, float32
3208     64 B    b2[16]     — layer-2 bias vector, float32
3272     512 B   W3[8][16]  — layer-3 weight matrix, row-major, float32
3784     32 B    b3[8]      — layer-3 bias vector, float32
──────────────────────────────────────────────────────
Total    3816 B
```

All numeric values are IEEE-754 single-precision, little-endian.  The
reader in `loadWeights()` verifies the magic and version before accepting
the file; a mismatch silently falls back to rule-based mode.

### Python export snippet

```python
import struct

MAGIC, VERSION = 0x4E435052, 1

def export_weights(model, path):
    """Export a trained CPRPolicy (torch.nn.Module) to the binary format."""
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    with open(path, "wb") as f:
        f.write(struct.pack("<II", MAGIC, VERSION))
        for key in ["net.0.weight", "net.0.bias",
                    "net.2.weight", "net.2.bias",
                    "net.4.weight", "net.4.bias"]:
            data = sd[key].flatten().tolist()
            f.write(struct.pack(f"<{len(data)}f", *data))
```

Layer order in `state_dict`: `net.0` = Linear(8→32), `net.2` = Linear(32→16),
`net.4` = Linear(16→8).  Weights are stored row-major as PyTorch exports them.

---

## Training pipeline

### Data collection

Run the simulator with each configuration across a set of benchmark decks,
recording `(features, config, wall_time)` tuples.  The existing
`--linear-solver` flags cover the full config space:

```bash
for conf in cprw cpr_trueimpes cpr_quasiimpes cpr_trueimpesanalytic; do
  flow --linear-solver=$conf --output-dir=sweep/$conf DECK.DATA
done
```

Feature logging can be enabled by setting the OPM log verbosity to debug
level and parsing the `NeuralCPRPolicy:` lines from the `.DBG` file (or by
instrumenting `ISTLSolver::extractFeatures()` to write a CSV).

### Model definition (PyTorch)

```python
import torch.nn as nn

class CPRPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8),
        )

    def forward(self, x):
        return self.net(x)   # raw logits, shape [B, 8]
```

### Loss and labels

Labels are the config index (per output head) whose wall-time-per-Newton-step
was minimum in the sweep.  Train each head independently with cross-entropy:

```python
# logits shape [B,8]; labels shape [B] each
loss = (
    F.cross_entropy(logits[:, 0:3], label_dec) +
    F.binary_cross_entropy_with_logits(logits[:, 3], label_cprw.float()) +
    F.cross_entropy(logits[:, 4:6], label_fine) +
    F.cross_entropy(logits[:, 6:8], label_coarse)
)
```

### Recommended sweep cases

| Case | Cells | Phases | Why |
|---|---|---|---|
| SPE1 | 300 | 3 | Baseline; fast to sweep many configs |
| SPE3 | 9,000 | 3 | Medium complexity |
| SPE9 | 9,000 | 3 | Heterogeneous permeability |
| NORNE_ATW2013 | 45,217 | 3 | Full field, diverse timestep difficulty |
