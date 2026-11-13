# Trained models

## `cpr_amg_policy_6feat_newton_x3.model`

Neural AMG policy with six inputs and no diagonal-dominance input.

At every CPR setup rebuild the model predicts the cost of the nine pressure-AMG
settings (prolongation damping 1.0 / 1.3 / 1.6 x coarsening target
1200 / 600 / 300). Flow leaves its default (1.0, 1200) only for a setting
predicted cheaper by more than `OPM_NEURAL_CPR_MIN_GAIN`.

### Inputs

Layout `0,1,20,3,4,7`, which flow selects for any six-input AMG model.

| Slot | Feature | Meaning |
|---|---|---|
| 0 | `dt_days` | time step length |
| 1 | `dt_ratio` | time step / previous time step |
| 20 | `nl_residual_norm` | global L2 norm of the right-hand side, log scale |
| 3 | `nl_residual_reduce` | that norm / its value at Newton iteration 0 |
| 4 | `nl_iteration` | Newton iteration index |
| 7 | `prev_linsolver_iters` | Krylov iterations of the previous solve |

None of these is read from the matrix. The model uses neither
`diag_dominance` (mean block-norm ratio over the first 200 rows of rank 0's
local matrix) nor `nnz_per_row` and `num_cells_log`, which change with the
number of MPI ranks.

### Training

- Network: 15 members averaged in one file (seeds 42, 100, 200, five bootstrap
  members each), each 6 → 32 → 32 → 9.
- Target: log cost per solve relative to the default setting (`--relative`),
  plus the log of the number of solves the setting's run needed over the
  number the default run needed in the surrounding report steps
  (`--newton-cost`). See the header of `train_cpr_amg_policy.py` for why the
  per-solve cost alone is not enough.
- Data: four AMG sweeps of Norne, SPE11A and SPE9 (nine settings per deck, 32
  ranks x 2 threads): `sweep_amg_20261015`, `sweep_amg_20261112`,
  `sweep_amg_20261112b`, `sweep_amg_20261113`; 12 996 reuse windows. The
  sweeps are not part of this repository.

```bash
cd doc/neural_cpr
OPENBLAS_NUM_THREADS=1 python3 train_cpr_amg_policy.py \
    --csvs {$S1,$S2,$S3,$S4}/{NORNE_ATW2013,SPE11A,SPE9}_pd{1.0,1.3,1.6}_ct{1200,600,300}.csv \
    --features 6 --relative --newton-cost --seeds 42 100 200 \
    --out models/cpr_amg_policy_6feat_newton_x3.model
```

With `S1`..`S4` the four sweep directories in the order above, this command
reproduces the file byte for byte (the order of the files matters).

### Running

```bash
export OPM_NEURAL_CPR_WEIGHTS=$PWD/doc/neural_cpr/models/cpr_amg_policy_6feat_newton_x3.model
export OPM_NEURAL_CPR_ACTIONS=amg
export OPM_NEURAL_CPR_MIN_GAIN=0.05
mpirun -np 32 --bind-to none flow --threads-per-process=2 DECK.DATA
```

No layout variable is needed. The model needs a flow built from this branch:
older builds have no slot 20 and no six-input layout, and would feed a
six-input model the first six slots without warning.

### Results

2026-11-13, 32 ranks x 2 threads, `--bind-to none`, one run per arm. SPE11B
was not in the training data.

| Deck | Arm | Linear solve (s) | Newton its | Krylov its |
|---|---|---|---|---|
| Norne | flow default | 23.3 | 1327 | 3062 |
| Norne | this model | 19.9 | 1363 | 3234 |
| SPE11A | flow default | 125.1 | 9866 | 56297 |
| SPE11A | this model | 93.4 | 8371 | 48719 |
| SPE11B | flow default | 115.0 | 3686 | 23577 |
| SPE11B | this model | 93.2 | 3647 | 20414 |

- These runs used a development build. A flow built from this branch gives the
  same Newton and Krylov counts and the same decisions on Norne (pd1.0/ct300
  37 times, pd1.0/ct600 9 times) and SPE11A.
- Iteration counts are reproducible; times are not. On SPE11B the first report
  step alone varies by tens of seconds between runs doing identical work, and
  this default run was a slow one. By work done the expected gain there is 6
  to 9 % of run time.
- On Norne and SPE11A the model matches earlier seven-input models that read
  `diag_dominance`; it does not improve on them.
- The model tells Norne from SPE11A mainly by the residual norm and the time
  step length, which depend on the scale of the problem. Three decks do not
  show how it behaves on a fourth.
