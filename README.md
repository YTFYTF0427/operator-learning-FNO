# neuropaths


Given a 2D domain `Omega` bounded by curves `x = phi(y)` (set A) and
`x = psi(y)` (set B), vertically periodic, with a divergence-free
turbulent drift `b`, learns the coefficient-to-solution map

    (b_1(x, y), b_2(x, y), [f^{-1}(x, y)])  ->  rho_react(x, y) = q^+(x, y) * q^-(x, y)

where `q^+`, `q^-` are the forward and backward committor functions
(probabilities of next/last visiting A vs B) and `rho_react` is the
reactive trajectory density. The pipeline is:

1. **Generate data**: solve the committor PDEs by finite differences on
   a 311×311 grid for many random divergence-free velocity fields,
   subsample to a coarse grid (16/32/63), write a CSV.
2. **Train**: fit `neuralop.models.FNO` to the (b → rho_react) map with
   the relative-L2 loss (`neuralop.LpLoss`).
3. **Evaluate**: zero-shot super-resolution at the other grid sizes
   (Table 1 of the dissertation).

## Layout

    src/neuropaths/
        pde/          # FD solvers, velocity fields, boundaries, transforms
        data/         # CSV generation + torch Dataset
        models/       # FNO builder around neuralop.models.FNO
        training/     # neuralop.Trainer wrapper + LpLoss
        evaluation/   # metrics + prediction plots
        config/       # dataclass ExperimentConfig + YAML loader
        cli/          # neuropaths-{generate,train,evaluate}
        utils/        # device selection, seeding, IO
    configs/          # experiment YAMLs (square_{16,32,63}, curved_32, smoke_test)
    slurm/            # SLURM job scripts for Warwick's Avon or Blythe clusters
    tests/            # pytest suite (PDE solvers + rejection sampling)

## Installation

Requires [uv](https://docs.astral.sh/uv/). The `neuraloperator` library is
declared as a dependency and resolved automatically.

To install uv on Avon, try
```bash
curl -LsSf https://astral.sh/uv/install.sh | env CARGO_HOME=~/.cargo sh
```
and then add to your `.bashrc`

    export PATH="~/.cargo/bin:$PATH"

Install the package with

```bash
git clone git@github.com:Tripudium/neuropaths.git
cd neuropaths
uv sync --frozen            # creates .venv and installs from uv.lock
source .venv/bin/activate   # not scrictly necessary
```

All commands run via `uv run <cmd>` so you don't need to activate the venv
explicitly. The reason we are doing it this way is because it seems more
convenient to have all the right packages available than having to use
what is available on Avon.

## Running locally

The pipeline is driven by a single YAML config that specifies the PDE
parameters, dataset size, model architecture, and training schedule.
See `configs/square_32.yaml` for the canonical setup (32×32 coarse
grid, 1000 train + 200 test PDE solutions, 100 epochs, FNO width 128).

For running on Avon or Blythe interactively, you might need to request 
and interactive shell, as outlined in 
[https://docs.scrtp.warwick.ac.uk/hpc-pages/hpc-interactive.html](https://docs.scrtp.warwick.ac.uk/hpc-pages/hpc-interactive.html)

### Generate data

```bash
uv run neuropaths-generate --config configs/square_32.yaml --split train --num-workers 8
uv run neuropaths-generate --config configs/square_32.yaml --split test  --num-workers 8
```

Outputs `runs/square_32/{train,test}.csv` with schema
`(solution_id, x, y, b1, b2, rho)`. `--num-workers` distributes the
PDE solves out across CPU cores via `multiprocessing.Pool`.

**Rejection sampling.** Solutions whose committor product `q^+ · q^-`
is essentially zero everywhere (no detectable transitions) carry no
learning signal. The generator
discards them: it draws `ceil(N · oversample_factor)` candidates and
keeps the first `N` whose `rho.max() ≥ rho_min_max`. Defaults are
`rho_min_max = 0.01` and `oversample_factor = 1.5`. Acceptance stats
are printed at the end of each split.

### Train

```bash
uv run neuropaths-train --config configs/square_32.yaml
```

Trains via `neuralop.Trainer` with the loss named in `cfg.train.loss`
(`relative_l2`, `absolute_l2`, `h1`, or `mse`) and Adam. Validation
relative-L2 is reported each epoch on `cfg.data.test_csv`. After the
best-val checkpoint is reloaded, the trainer prints a final summary:

    Final val metrics (best-checkpoint weights):
      per_sample_mean_rel_l2   = ...   ← what neuralop reports each epoch
      global_rel_l2            = ...   ← dissertation FNO_1_Test.py:102
      mse                      = ...
      mae                      = ...

Both rel-L2 numbers are useful: the per-sample mean averages
`||p_i - y_i||₂ / ||y_i||₂` over the val set (dominated by samples
with small `||y_i||₂`), while the global rel L2 flattens all
predictions and targets into one long vector and divides one big norm
by another (weights samples by `||y_i||₂`, so it tracks bulk-of-distribution
quality). When the targets vary by orders of magnitude in scale, the two
numbers can disagree by ~50%, and only the global metric is comparable
to the dissertation's Table 1 entries.

Checkpoints are written through neuralop's `BaseModel.save_checkpoint`,
which produces two files at the path prefix:

    runs/square_32/fno2d_state_dict.pt   # clean state_dict
    runs/square_32/fno2d_metadata.pkl    # init kwargs (model can be rebuilt without the YAML)

Reload with:

```python
from pathlib import Path
from neuralop.models import FNO

ckpt = Path("runs/square_32/fno2d.pth")
model = FNO.from_checkpoint(ckpt.parent, ckpt.stem, map_location="cuda")
```

### Available configs

- `configs/smoke_test.yaml` — 20 train + 5 test, 5 epochs, FNO width 64.
  Runs quickly, just to check everything works.
- `configs/smoke_test_curved.yaml` — same idea but with random
  trig-polynomial boundaries and `f^{-1}` as a third input channel.
  Use this to iterate on curved-domain code without the cost of a full
  curved run.
- `configs/square_{16,32,63}.yaml` — resolution sweep for Table 1 of
  the dissertation (Section 5.1). Default `velocity_kmax = 8` (laminar).
- `configs/square_32_turbulent.yaml` — sibling of `square_32.yaml`
  with `velocity_kmax = 16` for the turbulent regime of
  `Operator_learning_chapter.pdf` §2.4. Same architecture and training
  schedule; outputs go to `runs/square_32_turbulent/` so artefacts
  don't collide.
- `configs/section2_laminar_32.yaml` — laminar variant (`velocity_kmax = 8`).
- `configs/curved_32.yaml` — 5000 PDEs with random trig-polynomial
  boundaries, `f^{-1}` fed as an extra input channel (Section 5.2).

### Evaluate (work-in-progress)

`neuropaths-evaluate --config <yaml>` is there as structure but not yet
ported (see `cli/evaluate.py`); for now, use the two helper scripts
described below to inspect a trained run.

### Pulling code over from Avon

For plotting you'd want to work locally, even if you do the training on Avon
(see below on that).
You can pull the logs, run summaries and checkpoints over using rsync:

```bash
rsync -avh --progress  username@avon:~/neuropaths/slurm/logs .
rsync -avh --progress  username@avon:~/neuropaths/runs .
```

### Plot the training curve

`scripts/plot_training_curve.py` parses the logs that
`neuralop.Trainer(verbose=True)` emits and writes a train/val loss PNG
on a log y-scale.

```bash
uv run python scripts/plot_training_curve.py slurm/logs/neuropaths-train-12345.out
# -> slurm/logs/neuropaths-train-12345.png  (next to the log)

uv run python scripts/plot_training_curve.py runs/square_32/train.log -o curve.png
```

It looks for lines like `[N] ... train_err=...` and `Eval: val_l2=...`
and ignores everything else, so passing a full SLURM `.out` file works
even though it has setup noise interleaved.

### Inference on a held-out PDE

`scripts/inference_demo.py` regenerates one PDE solution with an
independent seed (so it cannot have leaked into train/test), runs it
through a checkpoint, and writes a 3-panel comparison PNG: the input
velocity magnitude `|b|`, the FD ground-truth `rho_react`, and the
FNO's prediction (with shared colour scale on the latter two).

For curved domains (`cfg.pde.domain == "curved"`) the panels are
warped from transformed coords back to the physical (x', y) domain
via `f^{-1}`, with the boundary curves φ(y) and ψ(y) overlaid in
black so the irregular shape is visible. For square domains the warp
is the identity and the panels are unit squares as before.

A complementary debugging tool, `scripts/plot_boundaries.py`, draws a
grid of random trig-polynomial domains so you can spot-check the
boundary generator without running a full PDE pipeline:

```bash
uv run python scripts/plot_boundaries.py --n 12 --seed 0 --output figs/boundaries.png
```

```bash
uv run python scripts/inference_demo.py --config configs/square_32.yaml
# -> runs/square_32/inference_demo.png

uv run python scripts/inference_demo.py \
    --config configs/square_32.yaml \
    --seed 1234 \
    --checkpoint runs/square_32/fno2d.pth \
    --output figs/demo_seed1234.png
```

The script prints per-sample relative L2 and mean absolute error so
you can compare to the train/val numbers from the training curve.

### Zero-shot super-resolution evaluation

`scripts/super_resolution_eval.py` reproduces the dissertation's
Table 1: train at one grid `s`, evaluate at every `s'` in
`cfg.eval.test_resolutions` without retraining. The FNO's
discretisation invariance means the same weights apply at any
`s' ≥ 2 · n_modes`; for `s' < 2 · n_modes` neuralop silently
truncates the spectral coefficients (so a `n_modes=12` model can be
evaluated at 16 — only 9 modes are usable, the rest are zeroed).

```bash
uv run python scripts/super_resolution_eval.py --config configs/square_32.yaml --num-workers 8
```

For each resolution it generates `runs/<name>/eval/super_res_test_<G>.csv`
on a fresh seed band (offset from train and val so draws are disjoint),
loads the train-time normalisation stats, runs the model, and prints:

```
   grid |     N |   global rel L2 |   per-sample mean |        MSE |        MAE
     16 |  1000 |          ...    |               ... |         ... |        ...
     32 |  1000 |          ...    |               ... |         ... |        ...
     63 |  1000 |          ...    |               ... |         ... |        ...
```

Rejection sampling is *disabled* during super-res generation so all
three resolutions see the same underlying PDEs — this makes the rows
directly comparable. The FD solve is the cost driver (~2 s/PDE at
fine_grid=311 on a single core), so locally use `--num-workers 8`.
Pass `--regenerate` to rebuild the CSVs (otherwise they're cached).

## Running on Avon / Blythe (SLURM)

`scrtp/` holds offline copies of Warwick's HPC cluster docs;
the SLURM scripts in `slurm/` follow those conventions. There are
four scripts:

| script | partition | resources | purpose |
|---|---|---|---|
| `slurm/smoke_test.slurm` | gpu | 1 L40, 10 CPUs, 30 min | smoke-test the pipeline end-to-end on `configs/smoke_test.yaml` (Blythe) |
| `slurm/generate_full.slurm` | (default CPU) | 168 CPUs (Blythe) / 48 CPUs (Avon), 2 h | generate train + test for a full run |
| `slurm/train_full.slurm` | gpu | 1 L40, 10 CPUs, 4 h | train on Blythe (Lovelace L40 GPU) |
| `slurm/train_full_avon.slurm` | gpu | 1 RTX 6000, 10 CPUs, 4 h | train on Avon (Quadro RTX 6000 GPU) |
| `slurm/super_resolution_eval.slurm` | gpu | 1 RTX 6000, 10 CPUs, 1 h | generate test data at all `cfg.eval.test_resolutions` and evaluate (Avon) |

The two training scripts differ only in their GPU specifications — Blythe
uses `--gres=gpu:lovelace_l40:1` with `--mem-per-cpu=5960`, Avon uses
`--gres=gpu:quadro_rtx_6000:1` with `--mem-per-cpu=4000` (4 GB/core,
matching the Avon spec). The
generate script is the same on both clusters, just adjust
`--cpus-per-task` to match the node size (168 on Blythe, 48 on Avon).

### One-time setup

1. Clone the repo somewhere you have write access. Anywhere works:
   `$HOME/neuropaths`, a project share, scratch — the SLURM scripts
   are in `${SLURM_SUBMIT_DIR}` (= the directory you `sbatch` from)
   so they're not tied to a specific path.
2. Activate the venv (virtual environment) once on a login node:

   ```bash
   cd path/to/neuropaths
   uv sync --frozen
   source .venv/bin/activate
   ```

3. Submit jobs from that same directory:

   ```bash
   cd path/to/neuropaths
   sbatch slurm/train_full.slurm
   ```

   Each script does `cd "${SLURM_SUBMIT_DIR}"` and activates `.venv/`
   relative to that, so subsequent runs start in seconds with no
   `uv sync` overhead per job.

If you ever change `pyproject.toml` or `uv.lock`, re-run `uv sync
--frozen` once to rebuild the venv before the next submission.


### Smoke test

Checks that the whole pipeline works on Avon or Blythe before you
commit to a long run.

```bash
sbatch slurm/smoke_test.slurm
```

Logs land in `slurm/logs/neuropaths-smoke-<jobid>.{out,err}`.

### Full run (chained generate and train)

Data generation parallelizes well across many CPUs
(each PDE solve is a few seconds, embarrassingly parallel), training
needs a GPU. You can submit them separately or as two jobs with a SLURM dependency so the
GPU node only allocates after generation succeeds:

```bash
gen=$(sbatch --parsable slurm/generate_full.slurm)
sbatch --dependency=afterok:$gen slurm/train_full.slurm
```

By default both scripts use `configs/square_32.yaml`. Override with
the `CONFIG` env var:

```bash
gen=$(CONFIG=configs/curved_32.yaml sbatch --parsable slurm/generate_full.slurm)
sbatch --dependency=afterok:$gen --export=ALL,CONFIG=configs/curved_32.yaml slurm/train_full.slurm
```

#### Laminar vs turbulent regime

The velocity-field Fourier-mode cap `pde.velocity_kmax` switches
between the two regimes of `Operator_learning_chapter.pdf` §2.4:

- `configs/square_32.yaml` — `velocity_kmax: 8` (laminar)
- `configs/square_32_turbulent.yaml` — `velocity_kmax: 16` (turbulent)

Same architecture and training schedule in both, only the velocity
spectrum differs. Output directories don't collide
(`runs/square_32/` vs `runs/square_32_turbulent/`), so you can chain
both runs back-to-back. On Avon:

```bash
# Laminar (the dissertation comparison baseline)
gen=$(CONFIG=configs/square_32.yaml sbatch --parsable slurm/generate_full.slurm)
sbatch --dependency=afterok:$gen --export=ALL,CONFIG=configs/square_32.yaml slurm/train_full_avon.slurm

# Turbulent
gen=$(CONFIG=configs/square_32_turbulent.yaml sbatch --parsable slurm/generate_full.slurm)
sbatch --dependency=afterok:$gen --export=ALL,CONFIG=configs/square_32_turbulent.yaml slurm/train_full_avon.slurm
```

Each pair takes ~25 min (generation, ~12000 PDEs at 2 s/solve on 48
cores) plus ~2 min (training, 50 epochs on the RTX 6000). Submit both
generate jobs immediately and they'll queue in parallel; the train
jobs only allocate the GPU once their respective generate completes.

Once you submitted jobs, you can check them with ```squeue``` or check the status with

```bash
sacct -u maskbg
```

There are two files in slurm/logs that record the progress and status for each run. You can look at these to see how things
are going.

## Status: square-domain baseline

`configs/square_32.yaml` (32×32 coarse grid, 5000 train + 1000 test,
50 epochs on a single Avon RTX 6000) reaches **global rel L2 = 0.401**
on the val set.

![inference example](runs/square_32/inference_demo.png)

Reproduce locally:

```bash
sbatch slurm/generate_full.slurm     # ~15 min on a 48-core Avon node
sbatch slurm/train_full_avon.slurm   # ~2 min on a Quadro RTX 6000
uv run python scripts/plot_training_curve.py slurm/logs/neuropaths-train-<jobid>.out
uv run python scripts/inference_demo.py --config configs/square_32.yaml
```

TODO

- Curved-domain experiment (`configs/curved_32.yaml`) needs the
  `f^{-1}` channel at inference and an `inverse_map` smoke test.
- Architectural knobs the fno-explained.pdf audit flagged that we did
  not enable: anisotropic modes `(8, 12)`, `norm="instance_norm"`,
  `channel_mlp_dropout`.
- Port `neuropaths-evaluate` so this table can be regenerated from a
  saved checkpoint without `_final_eval` being inlined in the trainer.

## Tests

```bash
uv run pytest                              # full suite
uv run pytest tests/test_pde_solvers.py    # FD solver tests
uv run pytest tests/test_data.py           # generation + rejection sampling
```

The PDE solver tests cover: analytic recovery on `b=0`
(`q^+(x,y) = x`), Dirichlet/periodic boundary correctness, the
discrete maximum principle on turbulent fields, and forward/backward
consistency under drift reversal.
