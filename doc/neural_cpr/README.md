# Neural CPR Policy

Three complementary learned policies that replace or augment static CPR
linear-solver settings with per-Newton-step decisions, trained from sweep
data collected on benchmark reservoir decks.

---

## Policies

| Policy | Controls | Key env var |
|---|---|---|
| **CPR variant** (neural-v3) | Which of 24 CPR configurations to use each Newton step | `OPM_NEURAL_CPR_WEIGHTS` |
| **Neural AMG** | Prolongation damping and coarsening target at each CPR setup rebuild | `OPM_NEURAL_CPR_WEIGHTS` + `OPM_NEURAL_CPR_ACTIONS=amg` |
| **Neural abort** | Whether to give up on a timestep attempt likely to be chopped | `OPM_CPR_ABORT_MODEL` |

---

## Architecture

### CPR variant

```
simulator state  ──┐
                   ├──► NeuralCPRFeatures (12–19 features) ──► NeuralCPRPolicy ──► CPRConfig (label 0–23)
assembled matrix ──┘

label = dec*8 + cprw*4 + fine*2 + coarse
  dec   : 0=quasiimpes  1=trueimpes  2=trueimpesanalytic
  cprw  : 0=cpr         1=cprw
  fine  : 0=paroverilu0 1=dilu
  coarse: 0=ilu0        1=dilu
```

MLP: `N → H → 24` (N = 12 core features, up to 19 with extended columns;
H = best of {8,12,16,24,32} from architecture search; depth 1 or 2).

### Neural AMG

MLP: `12 → H → 9` (3 dampings × 3 coarsening targets). Outputs are negated
log costs; argmax picks the cheapest predicted combination.

### Neural abort

MLP: `14 → H → 1` (sigmoid). Predicts P(chop) at each Newton iteration.
Input standardisation is folded into the first layer at export time.

---

## Files changed / added

| File | Role |
|---|---|
| `opm/simulators/linalg/NeuralCprFeatures.hpp` | Feature struct + normalisation (12–19 features) |
| `opm/simulators/linalg/NeuralCprPolicy.hpp` | CPR variant MLP, AMG policy, rule-based fallback, PropertyTree builder |
| `opm/simulators/linalg/NeuralAbortPolicy.hpp` | Early-abort policy |
| `opm/simulators/linalg/ISTLSolver.hpp` | Integration hooks + three-tier caching |
| `CMakeLists_files.cmake` | Header registration |

Detailed documentation:

- [NeuralCPRFeatures](NeuralCPRFeatures.md)
- [NeuralCPRPolicy](NeuralCPRPolicy.md)
- [ISTLSolver integration](ISTLSolver_integration.md)
- [Continual learning](CONTINUAL_LEARNING_README.md)

---

## Quick start

### CPR variant policy

```bash
export OPM_NEURAL_CPR_WEIGHTS=/path/to/cpr_policy.model
flow DECK.DATA
```

### Neural AMG policy (on Flow's default CPR reuse)

```bash
export OPM_NEURAL_CPR_WEIGHTS=/path/to/cpr_amg_policy.model
export OPM_NEURAL_CPR_ACTIONS=amg
export OPM_NEURAL_CPR_MIN_GAIN=0.05
flow DECK.DATA
```

### Neural abort policy

```bash
export OPM_CPR_ABORT_MODEL=/path/to/abort.model
export OPM_CPR_ABORT_THRESHOLD=0.95
flow DECK.DATA
```

### All three combined

```bash
export OPM_NEURAL_CPR_WEIGHTS=/path/to/cpr_policy.model
export OPM_CPR_ABORT_MODEL=/path/to/abort.model
export OPM_CPR_ABORT_THRESHOLD=0.95
flow DECK.DATA
```

### CPR variant with setup reuse

```bash
export OPM_NEURAL_CPR_WEIGHTS=/path/to/cpr_policy.model
flow --cpr-reuse-setup=4 --cpr-reuse-interval=10 DECK.DATA
```

### Rule-based fallback (no model file, zero overhead)

```bash
flow DECK.DATA
```

The `.DBG` log will show one line per solver rebuild:

```
NeuralCPRPolicy: active (rule-based fallback)
NeuralCPRPolicy: cprw weight=quasiimpes fine=paroverilu0 coarse=ilu0
```

---

## Training pipeline

```
# Step 0 — collect sweep data (hours of flow runs)
python run_sweep.py --flow ./flow --out cpr_sweep_data/ \
    --decks norne/NORNE.DATA spe9/SPE9.DATA

# Step 1 — label: pick best config per solve
python collect_sweep_labels.py \
    --csvs cpr_sweep_data/**/*.csv --out training_data.csv

# Step 2 — (optional) pre-train encoder
python pretrain_cpr_policy.py \
    --csvs cpr_sweep_data/**/*.csv --out encoder.npz --hidden 24

# Step 2 — train CPR variant policy
python train_cpr_policy.py \
    --data training_data.csv --out cpr_policy.model \
    --hidden 32 --epochs 200 --pretrained encoder.npz

# AMG policy (separate sweep with run_amg_sweep.py)
python train_cpr_amg_policy.py \
    --csvs amg_sweep/**/*.csv --out cpr_amg_policy.model

# Abort policy (from sweep .DBG logs)
python train_abort_policy.py \
    --train sweep_run1/**/*.DBG --out abort.model
```

All training scripts are pure NumPy; no PyTorch or TensorFlow required.
Continual learning (EWC + replay buffers) is available in all three scripts —
see [CONTINUAL_LEARNING_README.md](CONTINUAL_LEARNING_README.md).

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `OPM_NEURAL_CPR_WEIGHTS` | unset | Path to CPR variant or AMG model |
| `OPM_NEURAL_CPR_ACTIONS` | `variant` | `variant` or `amg` |
| `OPM_NEURAL_CPR_MIN_GAIN` | `0.05` | AMG policy: min log-cost gain to switch |
| `OPM_NEURAL_CPR_CONFIDENCE` | `0.6` | CPR variant: softmax confidence to switch |
| `OPM_NEURAL_CPR_FORBID_LABELS` | unset | Comma-separated labels to exclude (e.g. `16,17,18,19,20,21,22,23` for SPE11) |
| `OPM_NEURAL_CPR_DISABLE` | unset | `1` to disable CPR variant (abort-only mode) |
| `OPM_CPR_ABORT_MODEL` | unset | Path to abort policy model |
| `OPM_CPR_ABORT_THRESHOLD` | `0.95` | Abort: probability threshold to give up |
| `OPM_CPR_SWEEP_LOG` | unset | Per-Newton CSV written during sweep data collection |
