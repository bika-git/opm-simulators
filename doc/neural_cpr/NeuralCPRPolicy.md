# Module: NeuralCPRPolicy

**File:** `opm/simulators/linalg/NeuralCprPolicy.hpp`

---

## Purpose

Three complementary neural policies adapt the CPR linear solver at runtime:

| Policy | Env var | What it controls |
|---|---|---|
| **CPR variant** (neural-v3) | `OPM_NEURAL_CPR_WEIGHTS` | Which of 24 CPR configurations (decoupling × smoother) to use each Newton step |
| **Neural AMG** | `OPM_NEURAL_CPR_WEIGHTS` + `OPM_NEURAL_CPR_ACTIONS=amg` | Prolongation damping and coarsening target of the pressure AMG at each CPR setup rebuild |
| **Neural abort** | `OPM_CPR_ABORT_MODEL` | Whether to give up on a timestep attempt that is likely to be chopped |

All three share the same binary model file format and are loaded by `Opm::ML::NNModel`.

---

## CPR variant policy

### Architecture

```
input   (N)  ──► Dense(N→H, ReLU) [× depth]
hidden  (H)  ──► Dense(H→24, linear)
logits  (24) ──► argmax → label 0–23
```

N is the number of features in the training CSV (12 core features, up to 19
with extended columns). H and depth are controlled by `--hidden` / `--depth`
in `train_cpr_policy.py` (defaults: H=32, depth=1). The benchmark uses the
best H from an architecture search over {8, 12, 16, 24, 32}.

Output label encoding (flat joint space):

```
label = dec*8 + cprw*4 + fine*2 + coarse

dec   : 0=quasiimpes  1=trueimpes  2=trueimpesanalytic
cprw  : 0=cpr         1=cprw
fine  : 0=paroverilu0 1=dilu
coarse: 0=ilu0        1=dilu
```

Labels 0–23 span all combinations; the argmax of the 24 logits gives the
predicted cheapest config. A softmax confidence check
(`OPM_NEURAL_CPR_CONFIDENCE`, default 0.6) gates the switch; labels can be
forbidden per deck with `OPM_NEURAL_CPR_FORBID_LABELS`.

### Rule-based fallback

When no weight file is loaded the policy uses a deterministic heuristic:

```
Newton iter 0 (first linearisation of a new timestep):
    → quasiimpes + cpr + paroverilu0 + ilu0
    Cheap weight computation; first step is structurally easy.

Slow convergence (iter > 0 and residual_reduce > 0.5) or many iters (≥ 4):
    → trueimpes + cprw + dilu + ilu0
    More accurate decoupling and robust smoother when Newton struggles.

Otherwise (iters 1–3, converging well):
    → quasiimpes + cprw + dilu + ilu0
```

---

## Neural AMG policy

Loaded when `OPM_NEURAL_CPR_ACTIONS=amg` is set alongside
`OPM_NEURAL_CPR_WEIGHTS`. Controls two AMG parameters chosen at each CPR
setup rebuild:

| Parameter | Options |
|---|---|
| Prolongation damping | 1.0 / 1.3 / 1.6 |
| Coarsening target | 1200 / 600 / 300 |

### Architecture

```
input   (12) ──► Dense(12→H, ReLU) [× depth]
hidden  (H)  ──► Dense(H→9, linear)   [default layout; 6 with --additive]
outputs (9)  ──► argmax → (damping index, target index)
```

Outputs are negated log costs (`−log ms`), so the argmax gives the predicted
cheapest combination. The default anchor (`OPM_NEURAL_CPR_MIN_GAIN`, default
0.05) keeps the policy on the default (1.0, 1200) unless another combination
is predicted at least 5% cheaper (in log-cost).

Trained by `train_cpr_amg_policy.py` from sweep CSVs produced by
`run_amg_sweep.py`. Uses an ensemble of 5 bootstrap members averaged at
export time.

---

## Neural abort policy

Loaded via `OPM_CPR_ABORT_MODEL`. At every Newton iteration of a timestep
attempt that has not converged yet, predicts the probability that the attempt
will be chopped. If `prob > OPM_CPR_ABORT_THRESHOLD` (default 0.95), Flow
gives up on the attempt early.

### Architecture

```
input   (14) ──► Dense(14→H, ReLU)
hidden  (H)  ──► Dense(H→1, linear)
output  (1)  ──► sigmoid → P(chop)
```

H defaults to 16. Input standardisation `(x − μ) / σ` is folded into the
first layer at export time, so Flow feeds raw features with no preprocessing.

The 14 features are (in order):

```
SOLVER_KNOWN (5):  iteration, after_chop, chop_streak, dt_change, prev_iters
CPR_FEATURES (4):  last_lin_its, max_lin_its, setup_age, rebuilt_now
RES_FEATURES (5):  log_res, res_vs_start, res_step, res_vs_best, res_frac_up
```

Trained by `train_abort_policy.py` from flow `.DBG` logs paired with sweep
CSVs (`OPM_CPR_SWEEP_LOG`).

---

## Binary model file format

All three policies use the same format, written by `export_binary_fallback`
in `train_cpr_policy.py`:

```
Offset  Size    Content
0       4 B     n_layers  (uint32, little-endian)

For each layer i = 0 … n_layers-1:
  0     4 B     layer type = 3  (LAYER_DENSE, uint32)
  4     4 B     in_dim  (uint32)
  8     4 B     out_dim (uint32)
  12    4 B     out_dim (uint32, repeated — Kerasify-compatible header)
  16    in_dim × out_dim × 4 B    W.T  (float32, row-major, transposed weights)
  …     out_dim × 4 B             b    (float32, bias)
  …     4 B     activation code  (uint32)
                  1 = linear  (last layer)
                  2 = relu    (hidden layers)
                  3 = softplus
                  6 = tanh
```

All numeric values are IEEE-754 single-precision, little-endian. There is no
magic number or version field; the reader in `Opm::ML::NNModel` identifies
the layout from `n_layers` and the per-layer dimension headers.

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `OPM_NEURAL_CPR_WEIGHTS` | unset | Path to CPR variant or AMG model file |
| `OPM_NEURAL_CPR_ACTIONS` | `variant` | `variant` = CPR config policy; `amg` = AMG parameter policy |
| `OPM_NEURAL_CPR_MIN_GAIN` | `0.05` | AMG policy: minimum log-cost gain to switch from default |
| `OPM_NEURAL_CPR_CONFIDENCE` | `0.6` | CPR variant: softmax confidence threshold to switch |
| `OPM_NEURAL_CPR_FORBID_LABELS` | unset | Comma-separated label indices to exclude (e.g. `16,17,18,19,20,21,22,23` for SPE11) |
| `OPM_NEURAL_CPR_DISABLE` | unset | Set to `1` to disable CPR variant policy (run abort-only) |
| `OPM_CPR_ABORT_MODEL` | unset | Path to abort policy model file |
| `OPM_CPR_ABORT_THRESHOLD` | `0.95` | Abort policy: probability threshold to give up on an attempt |
| `OPM_CPR_SWEEP_LOG` | unset | Path for per-Newton-step solver CSV written during sweep runs |

---

## Training pipeline

```
Step 0  run_sweep.py          → sweep CSVs (24 configs × decks × runs)
Step 1  collect_sweep_labels.py → labelled training CSV (best config per solve)
Step 2  pretrain_cpr_policy.py  → encoder .npz (optional warm-start)
        train_cpr_policy.py     → cpr_policy.model  (CPR variant)

Separately:
        run_amg_sweep.py        → AMG sweep CSVs
        train_cpr_amg_policy.py → cpr_amg_policy.model
        train_abort_policy.py   → abort.model  (from sweep .DBG logs)
```

All training scripts are pure NumPy with a hand-written AdamW optimiser.
No PyTorch or TensorFlow dependency. Continual learning (EWC + replay
buffers) is supported by all three via `continual.py`.
