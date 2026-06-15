# FMint-SDE Reproduction Tutorial

This directory contains additive scripts for recreating the FMint-SDE paper experiments without modifying the existing repository code.

The existing `run.py` still writes checkpoints and tensorboard results under:

```bash
/export/nnoorani/FMint_SDE_Coloured_Noise/save/${RUN_USER}/...
```

Before launching real training jobs, make sure that path exists on your cluster, or create a symlink from that path to your preferred storage location. The scripts here do not patch `run.py`.

## Setup

From the repo root:

```bash
cp tutorials/fmint_sde_reproduction/fmint_sde_reproduction.env.example \
   tutorials/fmint_sde_reproduction/fmint_sde_reproduction.env
```

Edit `tutorials/fmint_sde_reproduction/fmint_sde_reproduction.env`:

```bash
export REPO_ROOT="/path/to/FMint_SDE_Coloured_Noise"
export DATA_ROOT="$REPO_ROOT/tutorials/fmint_sde_reproduction/data"
export LOG_ROOT="$REPO_ROOT/tutorials/fmint_sde_reproduction/logs"
export ANALYSIS_ROOT="$REPO_ROOT/tutorials/fmint_sde_reproduction/results"
export GPU="0"
export HF_HOME="/path/to/huggingface/cache"
```

For fine-tuning or few-shot evaluation, also set pretrained checkpoint stamps or directories:

```bash
export PRETRAIN_CAP_STAMP="YYYYMMDD-HHMMSS"
export PRETRAIN_NOCAP_STAMP="YYYYMMDD-HHMMSS"
export PRETRAIN_RESTORE_STEP="1000000"
```

For fine-tuned evaluation, use per-system stamps when each system was fine-tuned separately:

```bash
export FT_CAP_STOCHASTIC_LORENZ_STAMP="YYYYMMDD-HHMMSS"
export FT_NOCAP_STOCHASTIC_LORENZ_STAMP="YYYYMMDD-HHMMSS"
export FT_RESTORE_STEP="2000"
```

System names are converted to uppercase environment keys by replacing punctuation with underscores.
For Experiment 5 sweeps, include the dataset size:

```bash
export FT_NOCAP_STOCHASTIC_LORENZ_N50_STAMP="YYYYMMDD-HHMMSS"
```

## Dry Runs And Smoke Checks

Print commands without running them:

```bash
bash tutorials/fmint_sde_reproduction/reproduce_fmint_sde.sh \
  --experiment exp1 --stage all --mode both --dry-run
```

Run a tiny command-generation smoke pass:

```bash
bash tutorials/fmint_sde_reproduction/reproduce_fmint_sde.sh \
  --experiment exp1 --stage data --mode nocap --smoke --dry-run
```

Parse the fixture log:

```bash
python3 tutorials/fmint_sde_reproduction/parse_fmint_sde_logs.py \
  --log-root tutorials/fmint_sde_reproduction/fixtures \
  --output-csv tutorials/fmint_sde_reproduction/results/fixture_summary.csv \
  --output-md tutorials/fmint_sde_reproduction/results/fixture_summary.md
```

Verify 500-step rollout chunking:

```bash
python3 tutorials/fmint_sde_reproduction/rollout_500.py \
  --check-only --steps 500 --window 50 \
  --output-csv tutorials/fmint_sde_reproduction/results/rollout_check.csv
```

The rollout check should print `ROLL_OUT_WINDOWS: 10`.

## Experiment Commands

Experiment 1, Table 1 in-distribution systems:

```bash
bash tutorials/fmint_sde_reproduction/reproduce_fmint_sde.sh \
  --experiment exp1 --stage all --mode both --train-steps 2000
```

This covers:

- `geombrownian_motion`
- `mueller_overdamped`
- `periodic_nonlinearoscillator`
- `stochastic_lorenz`

The parser maps `strong` to AMD and `weak` to MAD. Coarse rows come from `analysis.py` output. Black-box surrogate and single-SDE rows are baseline hooks only; set `BASELINE_CSV` to merge external rows.

Experiment 2, computational efficiency:

```bash
bash tutorials/fmint_sde_reproduction/reproduce_fmint_sde.sh \
  --experiment exp2 --stage all --mode nocap
```

This generates/times stochastic Lorenz with 25 initial conditions, 40 noise realizations, fine `dt=1e-5`, and coarse `k*dt=1e-3`. The timing helper reports fine, coarse, and FMint-SDE runtime normalized to fine runtime.

Experiment 3, OOD transfer:

```bash
bash tutorials/fmint_sde_reproduction/reproduce_fmint_sde.sh \
  --experiment exp3 --stage all --mode both --train-steps 2000
```

Default systems are OU, inhomogeneous OU, double-well, coupled double-well, `duffing_langevin`, perturbed nonlinear oscillator, predator-prey, and fluxgate sensor.

Experiment 4, 500-step rollout:

```bash
bash tutorials/fmint_sde_reproduction/reproduce_fmint_sde.sh \
  --experiment exp4 --stage evaluate
```

By default this runs the synthetic chunking check. To score exported rollout arrays, create an `.npz` with:

- `coarse`: shape `(samples, 500, dim)`
- `fine`: shape `(samples, 500, dim)`
- `correction`: shape `(samples, 500, dim)`, predicted fine-minus-coarse correction

Then set:

```bash
export ROLLOUT_NPZ="/path/to/rollout_predictions.npz"
```

Experiment 5, fine-tuning data-size sensitivity:

```bash
bash tutorials/fmint_sde_reproduction/reproduce_fmint_sde.sh \
  --experiment exp5 --stage all --mode nocap --train-steps 2000
```

The default sweep is `N={5,50,100,250,500,1000,5000}`. Override with:

```bash
export DATA_SIZE_SWEEP="5 50 100"
```

Experiment 6, robustness regimes:

```bash
bash tutorials/fmint_sde_reproduction/reproduce_fmint_sde.sh \
  --experiment exp6 --stage finetune --mode nocap
```

This expects existing regime data under:

```bash
$DATA_ROOT/SDE_ft_diff_param_Aug12/<regime_name>
```

The runner fails clearly if a regime folder is missing.

Experiment 7, multi-modal caption ablation:

```bash
bash tutorials/fmint_sde_reproduction/reproduce_fmint_sde.sh \
  --experiment exp7 --stage all --mode both
```

This trains caption and no-caption models from scratch for 50 epochs on 250 parameter settings, then evaluates zero-shot with `K=0..4` using `--sweep_demo_nums`.
The data stage also creates the `SDE_ft_50/<system>` test folders used by this evaluation.

## Slurm

Single experiment:

```bash
sbatch tutorials/fmint_sde_reproduction/slurm_fmint_sde.sbatch exp1 all both
```

One Slurm array task per experiment:

```bash
sbatch --array=1-7 tutorials/fmint_sde_reproduction/slurm_fmint_sde.sbatch
```

## Outputs

Logs go to:

```bash
$LOG_ROOT/<experiment>/
```

Parsed summaries go to:

```bash
$ANALYSIS_ROOT/<experiment>/summary.csv
$ANALYSIS_ROOT/<experiment>/summary.md
```

For Table 1, use columns:

- `source=fmint`, `scope=summary` for FMint-SDE rows
- `source=coarse`, `scope=summary` for coarse rows
- `amd` for paper AMD
- `mad` for paper MAD
