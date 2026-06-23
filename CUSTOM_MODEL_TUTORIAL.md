# Custom Model Tutorial

This is a hands-on tutorial for the current JAX `icon_lm` path in this repository.

It does two things at the same time:

1. It shows what you should run from the terminal, in order.
2. It explains what the key files do underneath, including what healthy output should look like.

This guide is centered on the existing `icon_lm` branch, not the `gpt2`, `deepo`, or `fno` branches.

One important repo fact up front: this repository does not appear to ship TFRecord training data in-tree. In practice you will either:

1. Point `run.py` at an external dataset directory you already have, or
2. Generate TFRecords first with `data_preparation/datagen_fmint_sde.py`.

Another practical note: the expected-output sections below are matched to the actual print statements and runtime structure in this codebase. I could not fully replay an end-to-end training run inside this repo as-is because there is no checked-in local TFRecord dataset to execute against.

Throughout the guide, anything that changes run-to-run, such as timestamps, losses, random captions, and parameter values, is shown as a pattern rather than an exact value.

---

## Part 1: Repo Map

### High-level flow

```text
data generation
  -> TFRecord files on disk
  -> run.py
  -> dataloader_fmint_SDE.DataProvider
  -> runner_jax.Runner_lm
  -> models_lm.build_network_fn(...)
  -> training / evaluation output
  -> optional analysis/analysis.py
```

### The files that matter first

If you want to understand the current `icon_lm` training path, read these first:

1. `run.py`
2. `run.sh` or `run-icon.sh`
3. `dataloader_fmint_SDE.py`
4. `runner_jax.py`
5. `models_lm.py`
6. `config_data/train_lm_config.json`
7. `config_data/test_lm_config.json`
8. `config_model/model_lm_config.json`
9. `analysis/analysis.py`

### What each file is responsible for

- `run.py`
  - Main training entrypoint.
  - Loads configs.
  - Expands dataset globs.
  - Builds the data pipeline.
  - Chooses the runner based on `--model`.
  - Runs training, loss printing, test error evaluation, plotting, and checkpoint saving.

- `run.sh` and `run-icon.sh`
  - Example launch scripts.
  - Good for learning the expected flags.
  - Not portable as-is because they point at specific `/export/...` paths.

- `dataloader_fmint_SDE.py`
  - Reads TFRecords.
  - Selects demonstrations, question trajectories, and captions.
  - Builds the `Data` object that every model consumes.

- `runner_jax.py`
  - Wraps JAX models for training and evaluation.
  - Defines `Runner_vanilla` for `icon`.
  - Defines `Runner_lm` for `icon_lm`.

- `models_lm.py`
  - Defines the current `icon_lm` architecture.
  - Builds caption-aware and no-caption forward functions.
  - Builds prediction functions for evaluation time.

- `analysis/analysis.py`
  - Loads a saved model.
  - Runs evaluation and convergence analysis.
  - Writes plots to `analysis_dir`.

---

## Part 2: What To Run, In Order

## Step 0: Decide where your data lives

Before running anything, decide which of these is true:

1. You already have a directory full of TFRecords whose names begin with `train` and `test`.
2. You need to generate that data first.

`run.py` expects you to pass a directory via `--train_data_dirs`, then it appends globs like `train*` and `test*`.

Healthy expectation:

```text
If your data directory is /path/to/data, run.py will expand:
train_file_names -> /path/to/data/train*
test_file_names  -> /path/to/data/test*
```

Broken sign:

```text
The directory exists, but there are no matching train* or test* TFRecord files.
```

---

## Step 1: Set up the environment

The current repo README uses a Conda-based setup.

```bash
conda create -n icon python=3.10 -y
conda activate icon

conda install -c conda-forge cudatoolkit=11.8
pip install tensorflow==2.15.0
conda install -c conda-forge cudnn=8.9.7.29
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

pip install -r env.txt
pip install jaxlib==0.4.23+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

What this does:

- Creates the Python environment.
- Installs TensorFlow, JAX, and the repo dependencies.
- Sets the library path TensorFlow expects for GPU detection.

Expected output:

```text
conda create / pip install output is long.
The healthy signal is that the commands finish without ImportError-related failures.
```

Quick verification command:

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')))"
```

Expected output:

```text
TensorFlow version: 2.15.0
Num GPUs Available: <some integer>
```

What can vary:

- On a GPU machine, the number of GPUs should match what TensorFlow can see.
- On a CPU-only machine, `0` is normal.

Common failure signs:

```text
ModuleNotFoundError: No module named ...
ImportError related to CUDA / cuDNN
TensorFlow import crashes before printing a version
```

---

## Step 2: Optionally generate TFRecord data

If you do not already have TFRecords, generate them first.

The generator entrypoint is:

- `data_preparation/datagen_fmint_sde.py`

The helper script is:

- `data_preparation/datagen_fmint_sde.sh`

### Example: generate a tiny train split

Run this from `data_preparation/`:

```bash
cd data_preparation

OUT_DIR=/absolute/path/to/generated_data

CUDA_VISIBLE_DEVICES=0 python datagen_fmint_sde.py \
  --dir "$OUT_DIR" \
  --caption_mode train \
  --name train \
  --eqns 2 \
  --quests 1 \
  --eqn_types ornstein_uhlenbeck \
  --length 5000 \
  --dt 0.001 \
  --nv_step 100 \
  --seed 100
```

### Example: generate a tiny test split

```bash
CUDA_VISIBLE_DEVICES=0 python datagen_fmint_sde.py \
  --dir "$OUT_DIR" \
  --caption_mode test \
  --name test \
  --eqns 1 \
  --quests 1 \
  --eqn_types ornstein_uhlenbeck \
  --length 5000 \
  --dt 0.001 \
  --nv_step 100 \
  --seed 200
```

What this does:

- Generates trajectories for one equation family.
- Writes TFRecords into `OUT_DIR`.
- Produces separate train and test files because the `--name` flag becomes the filename prefix.

Expected output:

```text
============================================================
STARTING DATA GENERATION
============================================================

caption_mode : train
name : train
dir : /absolute/path/to/generated_data
eqn_types : ['ornstein_uhlenbeck']
...
Generating Ornstein-Uhlenbeck data...
===========/absolute/path/to/generated_data/train_ornstein_uhlenbeck_100_40.tfrecord===========
-------------------------------------------------- 1 --------------------------------------------------
equation: ornstein_uhlenbeck_params_..._nv_step_100
cond_k.shape: (...), cond_v.shape: (...), qoi_k.shape: (...), qoi_v.shape: (...)
...
Ornstein-Uhlenbeck generation completed in <seconds> seconds
```

Then verify files exist:

```bash
ls "$OUT_DIR"
```

Expected output:

```text
train_ornstein_uhlenbeck_100_40.tfrecord
test_ornstein_uhlenbeck_100_40.tfrecord
```

What can vary:

- Exact filename suffixes depend on `nv_step` and `num_repeat`.
- Exact equation parameter values will vary.

Common failure signs:

```text
FileNotFoundError
JAX / TensorFlow import failure
NaN found!
Exceeded maximum number of retries while generating a valid batch.
```

---

## Step 3: Run a tiny training smoke test

Use `run.py` directly first. Do not start with the shell scripts unless you have already edited their hard-coded paths.

For a first smoke test, start with:

- `--model icon_lm`
- `--backend jax`
- `--loss_mode nocap`
- `--epochs 1`
- `--steps_per_epoch 1`
- `--train_batch_size 2`
- `--plot_num 0`
- `--loss_freq 1`

Why `nocap` first:

- It is the simplest path.
- `run.py` will set `model_config['caption_len'] = 0` and clear caption loading when `cap` is not in `loss_mode`.
- That makes shape debugging much easier.

### Example smoke-test command

Run this from the repo root:

```bash
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

DATA_DIR=/absolute/path/to/generated_data

CUDA_VISIBLE_DEVICES=0 python run.py \
  --backend jax \
  --problem icon_lm_smoke \
  --model icon_lm \
  --epochs 1 \
  --steps_per_epoch 1 \
  --train_batch_size 2 \
  --train_data_dirs "$DATA_DIR" \
  --model_config_filename model_lm_config.json \
  --train_config_filename train_lm_config.json \
  --test_config_filename test_lm_config.json \
  --train_data_globs train* \
  --test_data_globs test* \
  --test_demo_num_list 1,3,5 \
  --loss_mode nocap \
  --seed 1 \
  --plot_num 0 \
  --loss_freq 1
```

What this command does:

- Uses the JAX `icon_lm` path.
- Loads train/test TFRecords.
- Builds one batch immediately so the model can infer shapes.
- Runs the tiny training loop.
- Prints training loss and test error almost immediately because `loss_freq=1`.

Expected output:

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

or, on CPU:

```text
[]
```

Then:

```text
tfboard : False
deterministic : True
user : user
problem : icon_lm_smoke
backend : jax
seed : 1
...
stamp: YYYYMMDD-HHMMSS
train_decay_steps = 1
train_warmup_steps = 0
train_file_names:
['/absolute/path/to/generated_data/train*']
test_file_names:
['/absolute/path/to/generated_data/test*']
train_config:
{...}
test_config:
{...}
-----------------------model config-----------------------
{...}
-----------------------model config end-----------------------
train_batch_size 2
test_batch_size 2
equation 0: ...
caption  0: ...
Data(...)
```

The shape print is especially important. On the JAX path, the data gets split by `num_devices`, so the batch usually looks like:

```text
input_id:         (num_devices, batch_per_device, caption_len) or (num_devices, batch_per_device, 0)
demo_cond_k:      (num_devices, batch_per_device, demo_num, demo_cond_len, k_dim)
demo_cond_v:      (num_devices, batch_per_device, demo_num, demo_cond_len, v_dim)
demo_cond_mask:   (num_devices, batch_per_device, demo_num, demo_cond_len)
demo_qoi_k:       (num_devices, batch_per_device, demo_num, demo_qoi_len, k_dim)
demo_qoi_v:       (num_devices, batch_per_device, demo_num, demo_qoi_len, v_dim)
demo_qoi_mask:    (num_devices, batch_per_device, demo_num, demo_qoi_len)
quest_cond_k:     (num_devices, batch_per_device, 1, quest_cond_len, k_dim)
quest_cond_v:     (num_devices, batch_per_device, 1, quest_cond_len, v_dim)
quest_cond_mask:  (num_devices, batch_per_device, 1, quest_cond_len)
quest_qoi_k:      (num_devices, batch_per_device, 1, quest_qoi_len, k_dim)
quest_qoi_mask:   (num_devices, batch_per_device, 1, quest_qoi_len)
label:            (num_devices, batch_per_device, 1, quest_qoi_len, v_dim)
```

Then model initialization output:

```text
<large Flax tabulation output>
<parameter tree print>
+++++++++++++++++++ train all variables ++++++++++++++++++++
+++++++++++++++++++ train without caption ++++++++++++++++++++
```

Then the first loss print:

```text
==================== step: 0, loss start ====================
train loss: <mean>+-<std>
test with demo num 1, no caption  , error: <mean>+-<std>
test with demo num 3, no caption  , error: <mean>+-<std>
test with demo num 5, no caption  , error: <mean>+-<std>
==================== step: 0, loss end ====================
```

What can vary:

- `num_devices` depends on `jax.devices()`.
- `caption` may be empty or dummy in `nocap` mode.
- Loss and error values will vary.
- The Flax tabulation can be very long.

Common failure signs:

```text
ValueError: model ... not supported
FileNotFoundError for dataset globs
Shape mismatch inside models_lm.py
JAX pmap errors caused by train_batch_size not dividing across devices
Tokenizer / caption errors if you try caption mode before nocap works
```

### One small runtime quirk to know

`run.py` loops over:

```python
for _ in range(FLAGS.epochs * FLAGS.steps_per_epoch + 1):
```

So the smoke test above performs a little more work than the flag values suggest. That is normal for this repo.

---

## Step 4: Run a longer training job

Once the smoke test works, scale up the exact same command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py \
  --backend jax \
  --problem icon_lm_pretrain \
  --model icon_lm \
  --epochs 100 \
  --steps_per_epoch 10000 \
  --train_batch_size 32 \
  --train_data_dirs /absolute/path/to/full_dataset \
  --model_config_filename model_lm_config.json \
  --train_config_filename train_lm_config.json \
  --test_config_filename test_lm_config.json \
  --train_data_globs train* \
  --test_data_globs test* \
  --test_demo_num_list 1,3,5 \
  --loss_mode nocap \
  --seed 1 \
  --vistest
```

What this does:

- Runs the same path at full scale.
- Uses multiple JAX devices if available.

Expected output:

```text
The same structure as the smoke test, but repeated for many more steps.
You should see loss blocks, test error blocks, and time estimates as training progresses.
```

Common failure signs:

```text
Out-of-memory errors
Very slow startup due to JAX compilation
Loss prints never appear because loss_freq is too large for your short run
```

---

## Step 5: Save checkpoints and TensorBoard logs

This repo has one important caveat:

- `run.py` only creates checkpoints when `--tfboard` is enabled.
- The save paths are hard-coded inside `run.py` to `/export/jyuan98/FMint_SDE/save/...`.

That means local checkpointing is not portable until you either:

1. Run on a machine where those paths exist, or
2. Edit the save path logic in `run.py`.

If you do enable it:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --backend jax \
  --problem icon_lm_ckpt \
  --model icon_lm \
  --epochs 5 \
  --steps_per_epoch 200 \
  --train_batch_size 32 \
  --train_data_dirs /absolute/path/to/full_dataset \
  --model_config_filename model_lm_config.json \
  --train_config_filename train_lm_config.json \
  --test_config_filename test_lm_config.json \
  --train_data_globs train* \
  --test_data_globs test* \
  --test_demo_num_list 1,3,5 \
  --loss_mode nocap \
  --seed 1 \
  --vistest \
  --tfboard \
  --save_freq 500
```

Expected output:

```text
current time: YYYYMMDD-HHMMSS
saved to /export/.../ckpts/... step <step_number>
```

Common failure signs:

```text
Permission denied creating /export/... directories
No such file or directory under /export/...
```

---

## Step 6: Fine-tune from a checkpoint

Fine-tuning uses the same `run.py` entrypoint plus:

- `--restore_dir`
- `--restore_step`

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --backend jax \
  --problem icon_lm_finetune \
  --model icon_lm \
  --epochs 5 \
  --steps_per_epoch 200 \
  --train_batch_size 32 \
  --train_data_dirs /absolute/path/to/finetune_dataset \
  --model_config_filename model_lm_config.json \
  --train_config_filename train_lm_config.json \
  --test_config_filename test_lm_config.json \
  --train_data_globs train* \
  --test_data_globs test* \
  --test_demo_num_list 1,3,5 \
  --loss_mode nocap \
  --seed 1 \
  --vistest \
  --restore_dir /absolute/path/to/checkpoints_or_export_path \
  --restore_step 1000
```

What this does:

- Builds the same model structure.
- Loads parameter values before training continues.

Expected output:

```text
restored params from /absolute/path/to/checkpoints_or_export_path, step 1000
```

Then the rest of the run should look like normal training.

Common failure signs:

```text
No such file: <restore_dir>/<restore_step>_params.pickle
Model config changed and old checkpoint no longer matches parameter names/shapes
```

---

## Step 7: Run analysis

The analysis script is in `analysis/analysis.py`.

It is easiest to run from inside the `analysis/` directory because the script uses relative paths like `../config_data/...`.

Example:

```bash
cd analysis

CUDA_VISIBLE_DEVICES=0 python analysis.py \
  --backend jax \
  --model icon_lm \
  --test_config_filename test_lm_precise_config.json \
  --model_config_filename model_lm_config.json \
  --test_data_dirs /absolute/path/to/finetune_dataset \
  --analysis_dir /absolute/path/to/analysis_output \
  --restore_dir /absolute/path/to/checkpoints_or_export_path \
  --restore_step 1000 \
  --batch_size 10 \
  --loss_mode nocap
```

What this does:

- Loads the model.
- Restores checkpoint parameters.
- Evaluates predictions and convergence behavior.
- Writes plots into `analysis_dir`.

Expected output:

```text
test_file_names:
['/absolute/path/to/finetune_dataset/test*']
==============data config==============
test_config:
{...}
==============data config end==============
-----------------------model config-----------------------
model_config:
{...}
-----------------------model config end-----------------------
restored params from /absolute/path/to/checkpoints_or_export_path, step 1000
...
equation=<equation_name>, num_demos=<n>: MAE=..., RMSE=..., strong=..., weak=...
strong: ...
weak: ...
```

Expected filesystem output:

```text
/absolute/path/to/analysis_output/*.pdf
```

Common failure signs:

```text
Running analysis.py from the wrong working directory and missing ../config_data paths
Checkpoint path mismatch
Batch-size / device-shape mismatch during evaluation
```

---

## Part 3: What The Code Does Under The Hood

This section maps the important runtime blocks to the console output you see.

## `run.py`

### Block A: config loading and path expansion

What it reads:

- `--train_data_dirs`
- `--test_data_dirs`
- `--train_data_globs`
- `--test_data_globs`
- `config_data/*.json`
- `config_model/*.json`

What it builds:

- `train_file_names`
- `test_file_names`
- `train_config`
- `test_config`
- `model_config`

Expected output:

```text
train_file_names:
['/path/to/data/train*']
test_file_names:
['/path/to/data/test*']
train_config:
{...}
test_config:
{...}
-----------------------model config-----------------------
{...}
-----------------------model config end-----------------------
```

Why this matters for a custom model:

- Your model will receive its hyperparameters entirely through `model_config`.
- If your custom model needs a new hyperparameter, it belongs in a new config file under `config_model/`.

### Block B: `loss_mode` preprocessing

What it does:

- If `'cap'` is not in `FLAGS.loss_mode`, `run.py` sets:
  - `model_config['caption_len'] = 0`
  - `train_config['load_list'] = []`
  - `test_config['load_list'] = []`

Expected output:

```text
The printed model config will show caption_len = 0.
The printed train/test configs will show load_list = [].
```

Why this matters for a custom model:

- Start your custom model with `nocap`.
- That removes caption complexity while you debug core sequence behavior.

### Block C: build the data pipeline

What it calls:

- `DataProvider(...)`
- `train_data.get_next_data(...)`

What it returns:

- `equation`
- `caption`
- `data`
- `label`

Expected output:

```text
equation 0: ...
caption  0: ...
Data(...)
```

Why this matters for a custom model:

- The `data` object is your true model input contract.
- `label` is the target your model must learn to predict.

### Block D: model dispatch

Current branches:

- `icon` -> `Runner_vanilla`
- `icon_lm` -> `Runner_lm`
- `gpt2` -> `runner_torch.Runner`
- `deepo`, `fno` -> `runner_deepo_torch.Runner`

Expected output:

```text
For icon_lm, you should see a Flax model tabulation and parameter tree print.
```

Why this matters for a custom model:

- To add `my_model`, you will add one more dispatch branch here.

### Block E: training loop

What it does:

- Runs `runner.iter(...)`
- Periodically calls `runner.get_loss(...)`
- Periodically calls `runner.get_error(...)`
- Optionally saves checkpoints and TensorBoard images

Expected output:

```text
==================== step: 0, loss start ====================
train loss: ...
test with demo num 1, no caption  , error: ...
...
==================== step: 0, loss end ====================
```

Why this matters for a custom model:

- Your custom model must support:
  - `runner.get_loss(...)`
  - `runner.get_pred(...)`
  - `runner.get_error(...)`

---

## `dataloader_fmint_SDE.py`

### The `Data` contract

This file defines the namedtuple your model receives:

```python
Data(
    input_id,
    embedding_raw,
    embedding_pool,
    embedding_mask,
    demo_cond_k,
    demo_cond_v,
    demo_cond_mask,
    demo_qoi_k,
    demo_qoi_v,
    demo_qoi_mask,
    quest_cond_k,
    quest_cond_v,
    quest_cond_mask,
    quest_qoi_k,
    quest_qoi_mask,
)
```

What `get_next_data(...)` returns:

```python
equation, caption, data, label
```

Expected runtime meaning:

- `equation`
  - equation identifier string
- `caption`
  - caption text or caption selection output
- `data`
  - structured model input
- `label`
  - target QOI values for the question trajectory

### `parse_function(...)`

What it does:

- Reads TFRecord fields:
  - `equation`
  - `caption`
  - `cond_k`
  - `cond_v`
  - `qoi_k`
  - `qoi_v`
- Optionally reads caption embeddings or token ids depending on `load_list`.

Expected output:

```text
No direct print by itself.
The effect shows up later as the shapes of data.input_id, data.embedding_mask, and the cond/qoi tensors.
```

Why this matters for a custom model:

- If you keep the TFRecord schema unchanged, you do not need to rewrite parsing.

### `select_demo_quest(...)` and `select_caption(...)`

What they do:

- Pick which demonstrations become support examples.
- Pick which trajectory becomes the question example.
- Pick which caption is used.

Expected output:

```text
Indirectly visible in:
- the printed caption
- the demo_num visible in tensor shapes
```

Why this matters for a custom model:

- Demo and question layout is already standardized before your model sees it.

### `build_sequence(...)`

What it does:

- Uses `data_sequence.build_others(...)`.
- Converts raw trajectory arrays into the fixed-length condition/QOI slices used by the model.

Expected output:

```text
length in use in build sequence: <integer>
```

Why this matters for a custom model:

- Your model sees the post-processed sequence form, not the raw full trajectory.

### `get_next_data(...)`

What it does:

- Pulls the next batch from `tf.data`.
- Converts tensors to NumPy.
- Optionally tokenizes captions in real time.
- Packs everything into `Data`.
- Splits the batch over JAX devices when `num_devices > 0`.

Expected output:

```text
No direct print inside get_next_data itself.
Its result is what run.py prints with print_eqn_caption(...) and tree.tree_map(...shape...).
```

Why this matters for a custom model:

- This is the exact input shape your model must accept.

### `split_data(...)`

What it does:

- Produces reduced-demo versions of the same batch for evaluation.
- This is how the repo computes test error for demo counts like `1,3,5`.

Expected output:

```text
test with demo num 1, ...
test with demo num 3, ...
test with demo num 5, ...
```

Why this matters for a custom model:

- Your prediction function must work for fewer demos at test time than at train time.

---

## `runner_jax.py`

### `Runner.init_fn(...)`

What it does:

- Stores devices.
- Builds optimizer state.
- Creates vmapped and pmapped predict/loss functions.
- Replicates parameters and optimizer state across devices.

Expected output:

```text
<parameter tree print>
+++++++++++++++++++ train all variables ++++++++++++++++++++
```

or:

```text
+++++++++++++++++++ train caption-related variables only ++++++++++++++++++++
```

Why this matters for a custom model:

- Your returned functions from `build_network_fn(...)` must be compatible with JAX `vmap` and `pmap`.

### `Runner_lm.__init__(...)`

What it expects from the model module:

```python
forward_with_caption_fn,
forward_without_caption_fn,
predict_with_caption_fn,
predict_without_caption_fn,
params
```

Expected output:

```text
If the import and return values are correct, model initialization proceeds normally.
If not, you will fail here immediately.
```

Why this matters for a custom model:

- This is the exact return contract your new `models_my_model.py` must satisfy.

### `_build_loss_fn(...)`

What it builds:

- caption loss
- no-caption loss
- optional consistency loss

Expected output:

Depending on `loss_mode`, you should see one of:

```text
+++++++++++++++++++ train with caption and without caption ++++++++++++++++++++
```

or:

```text
+++++++++++++++++++ train without caption ++++++++++++++++++++
```

or:

```text
+++++++++++++++++++ train with consistency loss ++++++++++++++++++++
```

Why this matters for a custom model:

- If you want to stay compatible with `Runner_lm`, your custom model must produce outputs with the same shape conventions these losses expect.

### `iter(...)`, `get_loss(...)`, `get_pred(...)`, `get_error(...)`

What they do:

- `iter(...)` updates parameters.
- `get_loss(...)` returns per-device, per-batch losses.
- `get_pred(...)` returns predictions.
- `get_error(...)` computes the evaluation metric used in the training loop.

Expected output:

```text
Visible indirectly in the step-wise loss and error prints from run.py.
```

Why this matters for a custom model:

- If your model wires correctly but `get_error(...)` fails, your predict path is not matching the expected question-QOI output shape.

---

## `models_lm.py`

### `build_matrices_from_data_shape(...)`

What it does:

- Builds the sequence mask structure for attention.
- Builds function-position indices.
- Builds the output mask that chooses which tokens count as outputs.

Expected output:

```text
No direct print.
Its correctness shows up when model initialization and forward passes succeed.
```

Why this matters for a custom model:

- If you reuse the same sequence format, this is the core “how attention is wired” helper.

### `IconGPTModel.setup(...)`

What it creates:

- pre-projection
- function-position embedding
- optional caption projection
- transformer
- output projection

Expected output:

```text
These modules appear in the Flax tabulation output during model initialization.
```

Why this matters for a custom model:

- This is the easiest place to swap the architecture while keeping the external training contract unchanged.

### `basic_forward(...)`

What it does:

- Builds the training sequence.
- Adds function-position embeddings.
- Optionally prepends caption tokens.
- Builds the attention mask.
- Runs the transformer.
- Projects outputs back to `out_dim`.

Expected output:

```text
No direct print.
If healthy, the next visible signal is that loss computation works.
```

Why this matters for a custom model:

- This is the most important method to understand before designing your own model.

### `__call__(...)`, `forward_without_caption(...)`, `predict(...)`

What they return:

- `__call__(...)`
  - training-time output with caption path
- `forward_without_caption(...)`
  - training-time output without caption path
- `predict(...)`
  - evaluation-time output for the question QOI only

Expected output shapes for one example:

```text
forward_with_caption_fn(...)      -> (num_training_targets, out_dim)
forward_without_caption_fn(...)   -> (num_training_targets, out_dim)
predict_with_caption_fn(...)      -> (quest_qoi_len, out_dim)
predict_without_caption_fn(...)   -> (quest_qoi_len, out_dim)
```

Why this matters for a custom model:

- These shapes are the heart of the contract `Runner_lm` assumes.

### `build_network_fn(...)`

What it does:

- Strips off device and batch axes with `x[0,0]`.
- Builds one-example masks and model parameters.
- Returns four callable functions plus `params`.

Expected output:

```text
<Flax tabulation output>
```

Why this matters for a custom model:

- Your replacement module should preserve this top-level function signature.

---

## Part 4: Build Your Own Model

This section documents the smallest set of changes needed to add a new JAX model that plugs into the existing `icon_lm` training path.

The goal is not to redesign the whole repo. The goal is to keep the current data format and runner logic, and only swap the model implementation.

## Step 1: Copy the model file

Example command:

```bash
cp models_lm.py models_my_model.py
```

What this does:

- Creates a new starting point for your model.

Expected output:

```text
cp prints nothing when it succeeds.
```

Quick verification:

```bash
ls models_my_model.py
```

Expected output:

```text
models_my_model.py
```

Why this matters:

- Start from a working contract, then change internals gradually.

## Step 2: Keep the `Data` input contract unchanged

Your new model should still accept the same `Data` structure from `dataloader_fmint_SDE.py`.

Do not change:

- `demo_cond_k`
- `demo_cond_v`
- `demo_cond_mask`
- `demo_qoi_k`
- `demo_qoi_v`
- `demo_qoi_mask`
- `quest_cond_k`
- `quest_cond_v`
- `quest_cond_mask`
- `quest_qoi_k`
- `quest_qoi_mask`

Expected output when this is correct:

```text
The shape print in run.py still looks normal before model creation.
```

Broken sign:

```text
AttributeError: 'Data' object has no attribute ...
```

## Step 3: Keep the `build_network_fn(...)` return contract unchanged

Your custom file should still export:

```python
def build_network_fn(data, key, config, return_model=False, compact=True, print_model=True):
    ...
    return (
        forward_with_caption_fn,
        forward_without_caption_fn,
        predict_with_caption_fn,
        predict_without_caption_fn,
        params,
    )
```

Expected output when this is correct:

```text
Runner_lm initializes without argument-count or unpacking errors.
```

Broken signs:

```text
ValueError: not enough values to unpack
TypeError about wrong function arguments
```

## Step 4: Add `my_model` dispatch in `runner_jax.py`

Minimal pattern:

```python
if model == 'icon_lm':
    import models_lm as models
elif model == 'my_model':
    import models_my_model as models
else:
    raise ValueError('model {} not implemented'.format(model))

forward_with_caption_fn, forward_without_caption_fn, self.predict_with_caption_fn, self.predict_without_caption_fn, self.params = \
    models.build_network_fn(data, next(self.rng), model_config, print_model=print_model)
```

Expected output when this is correct:

```text
Running with --model my_model no longer fails inside runner_jax.py.
```

Broken sign before the fix:

```text
ValueError: model my_model not implemented
```

## Step 5: Add `my_model` dispatch in `run.py`

Minimal pattern:

```python
elif FLAGS.model in ['icon_lm', 'my_model']:
    from runner_jax import Runner_lm
    runner = Runner_lm(
        seed=FLAGS.seed,
        model=FLAGS.model,
        data=data,
        model_config=model_config,
        optimizer=optimizer,
        trainable_mode=FLAGS.trainable_mode,
        loss_mode=FLAGS.loss_mode,
    )
```

Expected output when this is correct:

```text
run.py accepts --model my_model and reaches model initialization.
```

Broken sign before the fix:

```text
ValueError: model my_model not supported
```

## Step 6: Add `config_model/model_my_model_config.json`

Start by copying the current config:

```bash
cp config_model/model_lm_config.json config_model/model_my_model_config.json
```

Expected output:

```text
cp prints nothing when it succeeds.
```

Then edit only the keys your custom architecture really needs.

Expected output when the config is wired correctly:

```text
run.py prints:
-----------------------model config-----------------------
{...your custom model config...}
-----------------------model config end-----------------------
```

Broken signs:

```text
KeyError for a missing hyperparameter
ValueError from your model code about an unsupported config field
```

## Step 7: Smoke-test your custom model

Use the same tiny command as before, but switch:

- `--model my_model`
- `--model_config_filename model_my_model_config.json`

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --backend jax \
  --problem my_model_smoke \
  --model my_model \
  --epochs 1 \
  --steps_per_epoch 1 \
  --train_batch_size 2 \
  --train_data_dirs /absolute/path/to/generated_data \
  --model_config_filename model_my_model_config.json \
  --train_config_filename train_lm_config.json \
  --test_config_filename test_lm_config.json \
  --train_data_globs train* \
  --test_data_globs test* \
  --test_demo_num_list 1,3,5 \
  --loss_mode nocap \
  --seed 1 \
  --plot_num 0 \
  --loss_freq 1
```

Expected output when the wiring is correct:

```text
All the usual run.py setup prints appear.
The model tabulation now reflects models_my_model.py instead of models_lm.py.
You do not see:
  ValueError: model my_model not supported
You do not see:
  ValueError: model my_model not implemented
You eventually reach:
==================== step: 0, loss start ====================
train loss: ...
```

Broken signs:

```text
ImportError: No module named models_my_model
Shape mismatch inside your custom forward pass
Loss computation fails because forward output shape is wrong
Predict path works differently from the training forward path
```

---

## Part 5: Debugging Checklist

Use this section as a fast “healthy vs broken” map.

## Missing data files

Healthy:

```text
train_file_names and test_file_names print the paths you expect.
Your data directory contains TFRecords whose names match train* and test*.
```

Broken:

```text
run.py prints the right directory, but there are no matching files there.
```

## Caption and tokenizer issues

Healthy:

```text
nocap mode sets caption_len = 0 and load_list = [].
Caption mode prints a real caption and initializes without tokenizer errors.
```

Broken:

```text
Missing Hugging Face tokenizer downloads
Caption markdown files not found under data_preparation/<caption_dir>/
Unexpected input_id / embedding_mask shape mismatches
```

## Shape mismatches

Healthy:

```text
The first printed batch shapes look consistent with:
(num_devices, batch_per_device, demo_num, length, dim)
for demo tensors, and
(num_devices, batch_per_device, 1, length, dim)
for question tensors.
```

Broken:

```text
Errors in models_lm.py during concatenation, masking, or output masking
Errors in runner_jax.py when computing loss against label
```

## JAX device splitting issues

Healthy:

```text
train_batch_size divides cleanly across jax.devices().
```

Broken:

```text
Batch reshape or pmap-related errors
Unexpected leading dimension mismatch after get_next_data(...)
```

Practical fix:

- Start with `CUDA_VISIBLE_DEVICES=0`.
- Use a tiny `train_batch_size` such as `2`.

## TensorBoard and checkpoint path issues

Healthy:

```text
If --tfboard is enabled and the save path exists, you see:
saved to /export/... step <n>
```

Broken:

```text
Permission denied
No such file or directory under /export/...
```

Practical fix:

- For local smoke tests, omit `--tfboard`.
- Before serious training, edit the hard-coded save path in `run.py` to a location you control.

---

## Final working mental model

If you remember only one thing, remember this:

```text
run.py does not build a model from raw files on disk.
It first asks DataProvider for one real batch.
That batch defines the shapes.
Those shapes are then used to initialize the model.
The runner wraps that model for train / predict / error.
```

That is why the fastest path to a custom model is:

1. Keep the `Data` contract unchanged.
2. Keep the `build_network_fn(...)` return contract unchanged.
3. Start with `nocap`.
4. Use a one-step smoke test before changing anything bigger.

If those four things hold, you can change the internal architecture much more safely.
