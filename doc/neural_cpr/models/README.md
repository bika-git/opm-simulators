# Trained models

## `cpr_amg_policy_7feat_allrows_newton_x3.model`

Neural AMG policy with seven inputs. Its diagonal-dominance input is taken
over all owned rows of all ranks.

At every CPR setup rebuild the model predicts the cost of the nine pressure-AMG
settings (prolongation damping 1.0 / 1.3 / 1.6 x coarsening target
1200 / 600 / 300). Flow leaves its default (1.0, 1200) only for a setting
predicted cheaper by more than `OPM_NEURAL_CPR_MIN_GAIN`.

### Inputs

Layout `0,1,20,3,4,19,7`, which flow on this branch selects for any
seven-input AMG model.

| Slot | Feature | Meaning |
|---|---|---|
| 0 | `dt_days` | time step length |
| 1 | `dt_ratio` | time step / previous time step |
| 20 | `nl_residual_norm` | global L2 norm of the right-hand side, log scale |
| 3 | `nl_residual_reduce` | that norm / its value at Newton iteration 0 |
| 4 | `nl_iteration` | Newton iteration index |
| 19 | `diag_dominance` | diagonal against off-diagonal block norms, all rows, log scale |
| 7 | `prev_linsolver_iters` | Krylov iterations of the previous solve |

### `diag_dominance` on this branch

    diag_dominance = (sum of ||A_ii||_F) / (sum of ||A_ij||_F, j != i)

Both sums run over every owned row of every rank and are added over MPI;
overlap rows are skipped, so no cell is counted twice. The value is recomputed
at Newton iteration 0 of each time step (one pass over the matrix and two
reductions). Slot 19 is `clip((log10(max(x, 0.1)) + 1) / 4)`.

| | Before | On this branch |
|---|---|---|
| Rows | first 200 of the local matrix | every owned row |
| Ranks | rank 0's value reached the model | all, summed over MPI |
| Statistic | mean of the per-row ratios | ratio of the two sums |
| Depends on rank count and partitioning | yes | no |

- SPE9 gives the same value on the first solve at 1, 8 and 32 ranks.
- Models that read slot 6 or 19 and were trained on the earlier probe (the
  12-input `cpr_amg_policy.model`, earlier seven-input models) are not valid
  on this branch. Sweep CSVs written with the earlier probe hold a different
  quantity in the `diag_dominance` column and cannot be used for these slots.
- The earlier hand-picked seven-input layout `0,1,2,3,4,7,10` now has to be
  given as `OPM_NEURAL_CPR_AMG_FEATURES`.

### Training

- Network: 15 members averaged in one file (seeds 42, 100, 200, five bootstrap
  members each), each 7 → 32 → 32 → 9.
- Target: log cost per solve relative to the default setting (`--relative`),
  plus the log of the number of solves the setting's run needed over the
  number the default run needed in the surrounding report steps
  (`--newton-cost`). See the header of `train_cpr_amg_policy.py` for why the
  per-solve cost alone is not enough.
- Data: the three AMG sweeps of Norne, SPE11A and SPE9 that recorded the
  all-row `diag_dominance` (nine settings per deck, 32 ranks x 2 threads):
  `sweep_amg_20261112`, `sweep_amg_20261112b`, `sweep_amg_20261113`; 9 747
  reuse windows. The sweeps are not part of this repository.

```bash
cd doc/neural_cpr
OPENBLAS_NUM_THREADS=1 python3 train_cpr_amg_policy.py \
    --csvs {$S1,$S2,$S3}/{NORNE_ATW2013,SPE11A,SPE9}_pd{1.0,1.3,1.6}_ct{1200,600,300}.csv \
    --features 7 --relative --newton-cost --seeds 42 100 200 \
    --out models/cpr_amg_policy_7feat_allrows_newton_x3.model
```

With `S1`..`S3` the three sweep directories in the order above, this command
reproduces the file byte for byte (the order of the files matters).

### Running

```bash
export OPM_NEURAL_CPR_WEIGHTS=$PWD/doc/neural_cpr/models/cpr_amg_policy_7feat_allrows_newton_x3.model
export OPM_NEURAL_CPR_ACTIONS=amg
export OPM_NEURAL_CPR_MIN_GAIN=0.05
mpirun -np 32 --bind-to none flow --threads-per-process=2 DECK.DATA
```

No layout variable is needed. The model needs a flow built from this branch:
other builds compute `diag_dominance` differently or map seven inputs to other
slots, and would feed the model wrong values without warning.

### Results

2026-11-13, 32 ranks x 2 threads, `--bind-to none`, one run per arm. SPE11B
was not in the training data.

| Deck | Arm | Linear solve (s) | Newton its | Krylov its |
|---|---|---|---|---|
| Norne | flow default | 23.3 | 1327 | 3062 |
| Norne | this model | 20.8 | 1363 | 3234 |
| SPE11A | flow default | 125.1 | 9866 | 56297 |
| SPE11A | this model | 89.3 | 8104 | 46257 |
| SPE11B | flow default | 115.0 | 3686 | 23577 |
| SPE11B | this model | 91.9 | 3550 | 19752 |

- These runs used a development build. A flow built from this branch gives the
  same Newton and Krylov counts and the same decisions on Norne (pd1.0/ct600
  28 times, pd1.0/ct300 18 times) and SPE11A, and logs the same
  `diag_dominance` values on Norne's default run.
- Iteration counts are reproducible; times are not. On SPE11B the first report
  step alone varies by tens of seconds between runs doing identical work, and
  this default run was a slow one. By work done the expected gain there is 6
  to 9 % of run time.
- On Norne the model does the same work as the six-input model without a
  diagonal input (branch `neural-amg-6feat`); on SPE11A and SPE11B the two
  differ by less than the spread between runs.
- `diag_dominance` is nearly the same on Norne and SPE11A (0.21 and 0.25 on
  the slot's scale). The model tells the two apart mainly by the residual norm
  and the time step length, which depend on the scale of the problem. Three
  decks do not show how it behaves on a fourth.
