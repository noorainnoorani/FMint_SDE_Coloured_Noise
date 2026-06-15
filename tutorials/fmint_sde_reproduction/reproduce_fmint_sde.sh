#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_FILE="$SCRIPT_DIR/fmint_sde_reproduction.env"
for ((i = 1; i <= $#; i++)); do
  if [[ "${!i}" == "--env" ]]; then
    j=$((i + 1))
    ENV_FILE="${!j:-}"
  fi
done

if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$ENV_FILE"
fi

REPO_ROOT="${REPO_ROOT:-$REPO_ROOT_DEFAULT}"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/data}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$SCRIPT_DIR/results}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU="${GPU:-0}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
RUN_USER="${RUN_USER:-user}"
PROBLEM="${PROBLEM:-icon_lm}"
CKPT_ROOT="${CKPT_ROOT:-/export/nnoorani/FMint_SDE_Coloured_Noise/save/${RUN_USER}/ckpts/${PROBLEM}}"
SEED="${SEED:-1}"
PRETRAIN_RESTORE_STEP="${PRETRAIN_RESTORE_STEP:-1000000}"
FT_RESTORE_STEP="${FT_RESTORE_STEP:-2000}"
DATA_SIZE_SWEEP="${DATA_SIZE_SWEEP:-5 50 100 250 500 1000 5000}"
BASELINE_CSV="${BASELINE_CSV:-}"

EXPERIMENT="exp1"
STAGE="all"
MODE="both"
TRAIN_STEPS="2000"
SYSTEMS_ARG=""
DRY_RUN=0
SMOKE=0

usage() {
  cat <<'USAGE'
Usage:
  reproduce_fmint_sde.sh [options]

Options:
  --experiment exp1|exp2|exp3|exp4|exp5|exp6|exp7
  --stage data|pretrain|finetune|evaluate|time|parse|all
  --mode cap|nocap|both
  --train-steps 1000|2000
  --systems "system_a,system_b"
  --env path/to/fmint_sde_reproduction.env
  --dry-run
  --smoke
  --help

Examples:
  bash tutorials/fmint_sde_reproduction/reproduce_fmint_sde.sh --experiment exp1 --stage all --mode both --dry-run
  bash tutorials/fmint_sde_reproduction/reproduce_fmint_sde.sh --experiment exp3 --stage finetune --mode nocap --train-steps 1000
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment) EXPERIMENT="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --train-steps) TRAIN_STEPS="$2"; shift 2 ;;
    --systems) SYSTEMS_ARG="$2"; shift 2 ;;
    --env) shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

case "$EXPERIMENT" in exp1|exp2|exp3|exp4|exp5|exp6|exp7) ;; *) echo "Invalid --experiment: $EXPERIMENT" >&2; exit 2 ;; esac
case "$STAGE" in data|pretrain|finetune|evaluate|time|parse|all) ;; *) echo "Invalid --stage: $STAGE" >&2; exit 2 ;; esac
case "$MODE" in cap|nocap|both) ;; *) echo "Invalid --mode: $MODE" >&2; exit 2 ;; esac
case "$TRAIN_STEPS" in 1000|2000) ;; *) echo "Invalid --train-steps: $TRAIN_STEPS" >&2; exit 2 ;; esac

log() { printf '[fmint-repro] %s\n' "$*"; }
die() { printf '[fmint-repro] ERROR: %s\n' "$*" >&2; exit 1; }

split_list() {
  printf '%s\n' "$1" | tr ',' ' '
}

mode_list() {
  if [[ "$MODE" == "both" ]]; then
    printf 'cap\nnocap\n'
  else
    printf '%s\n' "$MODE"
  fi
}

default_systems() {
  case "$EXPERIMENT" in
    exp1) printf '%s\n' geombrownian_motion mueller_overdamped periodic_nonlinearoscillator stochastic_lorenz ;;
    exp2) printf '%s\n' stochastic_lorenz ;;
    exp3) printf '%s\n' ornstein_uhlenbeck inhomogeneous_ornsteinuhlenbeck double_well coupled_doublewell duffing_langevin perturbed_nonlinearoscillator predator_prey fluxgate_sensor ;;
    exp4) printf '%s\n' stochastic_lorenz ;;
    exp5) printf '%s\n' ornstein_uhlenbeck inhomogeneous_ornsteinuhlenbeck double_well coupled_doublewell duffing_langevin perturbed_nonlinearoscillator predator_prey fluxgate_sensor ;;
    exp6) printf '%s\n' duffing_langevin_noise_induced duffing_langevin_overdamped duffing_langevin_stochastic_resonance predator_prey_s0g4 predator_prey_s2g4 predator_prey_s2g6 stochastic_lorenz_chaotic stochastic_lorenz_dispersion stochastic_lorenz_spiral stochastic_lorenz_rho1 stochastic_lorenz_rho13.926 stochastic_lorenz_rho20 stochastic_lorenz_rho24.5 stochastic_lorenz_rho24.06 stochastic_lorenz_rho24.76 ;;
    exp7) printf '%s\n' ornstein_uhlenbeck double_well coupled_doublewell mueller_overdamped duffing_langevin perturbed_nonlinearoscillator periodic_nonlinearoscillator geombrownian_motion inhomogeneous_ornsteinuhlenbeck fluxgate_sensor stochastic_lorenz predator_prey ;;
  esac
}

selected_systems() {
  if [[ -n "$SYSTEMS_ARG" ]]; then
    split_list "$SYSTEMS_ARG"
  else
    default_systems
  fi
}

system_spec() {
  case "$1" in
    ornstein_uhlenbeck) printf '5000 0.001 100 100\n' ;;
    double_well) printf '10000 1e-5 100 101\n' ;;
    coupled_doublewell) printf '10000 1e-5 100 102\n' ;;
    mueller_overdamped) printf '5000 1e-5 100 103\n' ;;
    duffing_langevin) printf '5000 1e-4 100 104\n' ;;
    perturbed_nonlinearoscillator) printf '20000 1e-5 100 102\n' ;;
    periodic_nonlinearoscillator) printf '2000 1e-5 10 102\n' ;;
    geombrownian_motion) printf '5000 0.0005 100 107\n' ;;
    inhomogeneous_ornsteinuhlenbeck) printf '20000 0.001 100 108\n' ;;
    fluxgate_sensor) printf '20000 1e-3 100 109\n' ;;
    stochastic_lorenz) printf '5000 1e-4 100 110\n' ;;
    predator_prey) printf '2000 0.005 10 111\n' ;;
    predator_prey2) printf '2000 0.005 10 111\n' ;;
    *)
      # Regime datasets in Experiment 6 are expected to already exist.
      printf '0 0 0 0\n'
      ;;
  esac
}

smoke_spec() {
  printf '500 1e-4 10 7\n'
}

upper_key() {
  printf '%s' "$1" | tr '[:lower:].-' '[:upper:]__' | sed 's/[^A-Z0-9_]/_/g'
}

run_logged() {
  local workdir="$1"
  local logfile="$2"
  shift 2
  mkdir -p "$(dirname "$logfile")"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'cd %q && HF_HOME=%q CUDA_VISIBLE_DEVICES=%q ' "$workdir" "$HF_HOME" "$GPU"
    printf '%q ' "$@"
    printf '> %q 2>&1\n' "$logfile"
  else
    log "Running: $logfile"
    (
      cd "$workdir"
      export HF_HOME
      export CUDA_VISIBLE_DEVICES="$GPU"
      "$@" > "$logfile" 2>&1
    )
  fi
}

restore_dir_for() {
  local phase="$1"
  local mode="$2"
  local system="${3:-}"
  local sample_count="${4:-}"
  local system_key
  system_key="$(upper_key "$system")"
  local mode_key
  mode_key="$(upper_key "$mode")"

  local candidates=()
  if [[ "$phase" == "pretrain" ]]; then
    candidates+=("PRETRAIN_${mode_key}_RESTORE_DIR")
    candidates+=("PRETRAIN_${mode_key}_STAMP")
  else
    if [[ -n "$sample_count" ]]; then
      candidates+=("FT_${mode_key}_${system_key}_N${sample_count}_RESTORE_DIR")
      candidates+=("FT_${mode_key}_${system_key}_N${sample_count}_STAMP")
    fi
    candidates+=("FT_${mode_key}_${system_key}_RESTORE_DIR")
    candidates+=("FT_${mode_key}_${system_key}_STAMP")
    candidates+=("FT_${mode_key}_RESTORE_DIR")
    candidates+=("FT_${mode_key}_STAMP")
  fi

  local var value
  for var in "${candidates[@]}"; do
    value="${!var:-}"
    if [[ -n "$value" ]]; then
      if [[ "$var" == *"_STAMP" ]]; then
        printf '%s/%s\n' "$CKPT_ROOT" "$value"
      else
        printf '%s\n' "$value"
      fi
      return 0
    fi
  done

  if [[ "$DRY_RUN" -eq 1 ]]; then
    local n_suffix=""
    [[ -n "$sample_count" ]] && n_suffix="_N${sample_count}"
    printf '<SET_%s_%s%s_RESTORE_DIR_FOR_%s>\n' "$(upper_key "$phase")" "$mode_key" "$n_suffix" "${system_key:-ALL}"
    return 0
  fi
  die "Missing restore dir/stamp for phase=$phase mode=$mode system=${system:-all}. Edit fmint_sde_reproduction.env."
}

ft_epochs() {
  printf '%d\n' "$((TRAIN_STEPS / 200))"
}

ft_data_dir() {
  local n="$1"
  local system="$2"
  printf '%s/SDE_ft_%s/%s\n' "$DATA_ROOT" "$n" "$system"
}

pretrain_data_dir() {
  printf '%s/caption-test-folder\n' "$DATA_ROOT"
}

analysis_log_dir() {
  printf '%s/%s/%s\n' "$LOG_ROOT" "$EXPERIMENT" "$1"
}

generate_ft_data() {
  local n="$1"
  local testquests="${2:-50}"
  local system length dt nv_step seed spec dir
  for system in $(selected_systems); do
    spec="$(system_spec "$system")"
    read -r length dt nv_step seed <<< "$spec"
    if [[ "$length" == "0" ]]; then
      [[ -d "$DATA_ROOT/SDE_ft_diff_param_Aug12/$system" ]] || die "Missing regime dataset folder for $system under $DATA_ROOT/SDE_ft_diff_param_Aug12"
      continue
    fi
    if [[ "$SMOKE" -eq 1 ]]; then
      read -r length dt nv_step seed <<< "$(smoke_spec)"
      n=2
      testquests=2
    fi
    if [[ "$EXPERIMENT" == "exp2" && "$system" == "stochastic_lorenz" ]]; then
      length=5000
      dt=1e-5
      nv_step=100
      testquests=25
    fi
    dir="$(ft_data_dir "$n" "$system")"
    run_logged "$REPO_ROOT/data_preparation" "$LOG_ROOT/$EXPERIMENT/data/${system}_N${n}.log" \
      "$PYTHON_BIN" datagen_fmint_sde_ft.py --dir "$dir" \
      --caption_mode train --name train --eqns 1 --quests "$n" --testquests "$testquests" \
      --eqn_types "$system" --length "$length" --dt "$dt" --nv_step "$nv_step" --seed "$seed"
    if [[ "$SMOKE" -eq 1 ]]; then
      break
    fi
  done
}

generate_pretrain_data() {
  local train_eqns=250
  local test_eqns=25
  local train_quests=1
  local test_quests=1
  local system length dt nv_step seed spec
  if [[ "$SMOKE" -eq 1 ]]; then
    train_eqns=1
    test_eqns=1
  fi
  for system in $(selected_systems); do
    spec="$(system_spec "$system")"
    read -r length dt nv_step seed <<< "$spec"
    [[ "$length" == "0" ]] && continue
    if [[ "$SMOKE" -eq 1 ]]; then
      read -r length dt nv_step seed <<< "$(smoke_spec)"
    fi
    run_logged "$REPO_ROOT/data_preparation" "$LOG_ROOT/$EXPERIMENT/data/pretrain_${system}_train.log" \
      "$PYTHON_BIN" datagen_fmint_sde.py --dir "$(pretrain_data_dir)" \
      --caption_mode train --name train --eqns "$train_eqns" --quests "$train_quests" \
      --eqn_types "$system" --length "$length" --dt "$dt" --nv_step "$nv_step" --seed "$seed"
    run_logged "$REPO_ROOT/data_preparation" "$LOG_ROOT/$EXPERIMENT/data/pretrain_${system}_test.log" \
      "$PYTHON_BIN" datagen_fmint_sde.py --dir "$(pretrain_data_dir)" \
      --caption_mode test --name test --eqns "$test_eqns" --quests "$test_quests" \
      --eqn_types "$system" --length "$length" --dt "$dt" --nv_step "$nv_step" --seed "$((seed + 100))"
    if [[ "$SMOKE" -eq 1 ]]; then
      break
    fi
  done
}

train_pretrain() {
  local mode="$1"
  local extra=()
  [[ "$mode" == "nocap" ]] && extra=(--loss_mode nocap)
  run_logged "$REPO_ROOT" "$LOG_ROOT/$EXPERIMENT/pretrain/${mode}.log" \
    "$PYTHON_BIN" run.py --problem "$PROBLEM" --epochs 50 \
    --train_batch_size 32 --train_data_dirs "$(pretrain_data_dir)" \
    --model_config_filename model_lm_config.json \
    --train_config_filename train_lm_config.json \
    --test_config_filename test_lm_config.json \
    --train_data_globs 'train*' --test_data_globs 'test*' \
    --test_demo_num_list 1,3,4 --model icon_lm \
    --nodeterministic --seed "$SEED" --vistest --tfboard "${extra[@]}"
}

train_finetune() {
  local mode="$1"
  local n="$2"
  local epochs batch_size restore_dir extra system data_dir
  epochs="$(ft_epochs)"
  restore_dir="$(restore_dir_for pretrain "$mode")"
  extra=()
  [[ "$mode" == "nocap" ]] && extra=(--loss_mode nocap)
  if [[ "$mode" == "nocap" ]]; then
    batch_size="${FT_NOCAP_BATCH_SIZE:-5}"
  else
    batch_size="${FT_CAP_BATCH_SIZE:-32}"
  fi
  for system in $(selected_systems); do
    if [[ "$EXPERIMENT" == "exp6" ]]; then
      data_dir="$DATA_ROOT/SDE_ft_diff_param_Aug12/$system"
      [[ "$DRY_RUN" -eq 1 || -d "$data_dir" ]] || die "Missing $data_dir"
    else
      data_dir="$(ft_data_dir "$n" "$system")"
    fi
    run_logged "$REPO_ROOT" "$LOG_ROOT/$EXPERIMENT/finetune/${mode}/${system}_N${n}_${TRAIN_STEPS}steps.log" \
      "$PYTHON_BIN" run.py --problem "$PROBLEM" --epochs "$epochs" \
      --train_batch_size "$batch_size" --train_data_dirs "$data_dir" \
      --restore_dir "$restore_dir" --restore_step "$PRETRAIN_RESTORE_STEP" \
      --steps_per_epoch 200 \
      --model_config_filename model_lm_config.json \
      --train_config_filename train_lm_config.json \
      --test_config_filename test_lm_config.json \
      --train_data_globs 'train*' --test_data_globs 'test*' \
      --test_demo_num_list 1,3,5 --model icon_lm \
      --nodeterministic --seed "$SEED" --vistest --tfboard \
      --save_freq "$TRAIN_STEPS" "${extra[@]}"
    if [[ "$SMOKE" -eq 1 ]]; then
      break
    fi
  done
}

evaluate_models() {
  local phase="$1"
  local mode="$2"
  local n="$3"
  local restore_dir restore_step extra system data_dir out_dir log_file
  extra=(--test_caption_id_list 0)
  if [[ "$mode" == "nocap" ]]; then
    extra=(--loss_mode nocap --test_caption_id_list -1)
  fi
  for system in $(selected_systems); do
    if [[ "$phase" == "fewshot" ]]; then
      restore_dir="$(restore_dir_for pretrain "$mode")"
      restore_step="$PRETRAIN_RESTORE_STEP"
    else
      restore_dir="$(restore_dir_for ft "$mode" "$system" "$n")"
      restore_step="$FT_RESTORE_STEP"
    fi
    if [[ "$EXPERIMENT" == "exp6" ]]; then
      data_dir="$DATA_ROOT/SDE_ft_diff_param_Aug12/$system"
    else
      data_dir="$(ft_data_dir "$n" "$system")"
    fi
    out_dir="$ANALYSIS_ROOT/$EXPERIMENT/$phase/$mode/N${n}/$system"
    log_file="$LOG_ROOT/$EXPERIMENT/evaluate/$phase/$mode/N${n}/${system}.log"
    run_logged "$REPO_ROOT/analysis" "$log_file" \
      "$PYTHON_BIN" analysis.py --correction --backend jax \
      --model icon_lm --test_config_filename test_lm_precise_config.json \
      --model_config_filename model_lm_config.json \
      --test_data_dirs "$data_dir" \
      --analysis_dir "$out_dir" \
      --restore_dir "$restore_dir" \
      --restore_step "$restore_step" \
      --batch_size "${ANALYSIS_BATCH_SIZE:-10}" "${extra[@]}"
    if [[ "$SMOKE" -eq 1 ]]; then
      break
    fi
  done
}

evaluate_exp7() {
  local mode="$1"
  local restore_dir extra system
  restore_dir="$(restore_dir_for pretrain "$mode")"
  extra=(--test_caption_id_list 0)
  [[ "$mode" == "nocap" ]] && extra=(--loss_mode nocap --test_caption_id_list -1)
  for system in $(selected_systems); do
    run_logged "$REPO_ROOT/analysis" "$LOG_ROOT/$EXPERIMENT/evaluate/zeroshot_sweep/$mode/${system}.log" \
      "$PYTHON_BIN" analysis.py --correction --backend jax \
      --model icon_lm --test_config_filename test_lm_precise_config.json \
      --model_config_filename model_lm_config.json \
      --test_data_dirs "$(ft_data_dir 50 "$system")" \
      --analysis_dir "$ANALYSIS_ROOT/$EXPERIMENT/zeroshot_sweep/$mode/$system" \
      --restore_dir "$restore_dir" --restore_step "$PRETRAIN_RESTORE_STEP" \
      --batch_size "${ANALYSIS_BATCH_SIZE:-10}" \
      --test_demo_num_list 0,1,2,3,4 --sweep_demo_nums "${extra[@]}"
    if [[ "$SMOKE" -eq 1 ]]; then
      break
    fi
  done
}

run_timing() {
  local timing_log="$LOG_ROOT/$EXPERIMENT/time/lorenz_efficiency.log"
  local analysis_logs="$LOG_ROOT/$EXPERIMENT/evaluate"
  run_logged "$REPO_ROOT" "$timing_log" \
    "$PYTHON_BIN" "$SCRIPT_DIR/time_lorenz_efficiency.py" \
    --initial-conditions 25 --noise-realizations 40 \
    --fine-dt 1e-5 --coarse-dt 1e-3 --coarse-steps 50 \
    --analysis-log-root "$analysis_logs" \
    --output-csv "$ANALYSIS_ROOT/$EXPERIMENT/lorenz_efficiency.csv"
}

run_rollout_500() {
  local npz="${ROLLOUT_NPZ:-}"
  local args=(--steps 500 --window 50 --output-csv "$ANALYSIS_ROOT/$EXPERIMENT/rollout_500.csv")
  if [[ -n "$npz" ]]; then
    args+=(--input-npz "$npz")
  else
    args+=(--check-only)
  fi
  run_logged "$REPO_ROOT" "$LOG_ROOT/$EXPERIMENT/rollout/rollout_500.log" \
    "$PYTHON_BIN" "$SCRIPT_DIR/rollout_500.py" "${args[@]}"
}

parse_logs() {
  local args=(--log-root "$LOG_ROOT/$EXPERIMENT" --output-csv "$ANALYSIS_ROOT/$EXPERIMENT/summary.csv" --output-md "$ANALYSIS_ROOT/$EXPERIMENT/summary.md")
  [[ -n "$BASELINE_CSV" ]] && args+=(--baseline-csv "$BASELINE_CSV")
  run_logged "$REPO_ROOT" "$LOG_ROOT/$EXPERIMENT/parse.log" \
    "$PYTHON_BIN" "$SCRIPT_DIR/parse_fmint_sde_logs.py" "${args[@]}"
}

stage_data() {
  case "$EXPERIMENT" in
    exp1|exp3) generate_ft_data 50 50 ;;
    exp2) generate_ft_data 50 25 ;;
    exp4) generate_ft_data 50 50 ;;
    exp5)
      local n
      for n in $DATA_SIZE_SWEEP; do
        generate_ft_data "$n" 50
      done
      ;;
    exp6)
      local system data_dir
      for system in $(selected_systems); do
        data_dir="$DATA_ROOT/SDE_ft_diff_param_Aug12/$system"
        [[ "$DRY_RUN" -eq 1 || -d "$data_dir" ]] || die "Expected regime data at $data_dir"
      done
      ;;
    exp7)
      generate_pretrain_data
      generate_ft_data 50 50
      ;;
  esac
}

stage_pretrain() {
  local mode
  for mode in $(mode_list); do
    train_pretrain "$mode"
  done
}

stage_finetune() {
  local mode n
  for mode in $(mode_list); do
    if [[ "$EXPERIMENT" == "exp5" ]]; then
      for n in $DATA_SIZE_SWEEP; do
        train_finetune "$mode" "$n"
      done
    else
      train_finetune "$mode" 50
    fi
  done
}

stage_evaluate() {
  local mode n
  if [[ "$EXPERIMENT" == "exp4" ]]; then
    run_rollout_500
    return
  fi
  for mode in $(mode_list); do
    if [[ "$EXPERIMENT" == "exp7" ]]; then
      evaluate_exp7 "$mode"
    elif [[ "$EXPERIMENT" == "exp5" ]]; then
      for n in $DATA_SIZE_SWEEP; do
        evaluate_models finetuned "$mode" "$n"
      done
    else
      evaluate_models fewshot "$mode" 50
      evaluate_models finetuned "$mode" 50
    fi
  done
}

main() {
  mkdir -p "$DATA_ROOT" "$LOG_ROOT" "$ANALYSIS_ROOT"
  if [[ "$SMOKE" -eq 1 ]]; then
    log "Smoke mode enabled: tiny data settings and first selected system only."
  fi
  case "$STAGE" in
    data) stage_data ;;
    pretrain) stage_pretrain ;;
    finetune) stage_finetune ;;
    evaluate) stage_evaluate ;;
    time) [[ "$EXPERIMENT" == "exp2" ]] && run_timing || die "--stage time is only defined for exp2" ;;
    parse) parse_logs ;;
    all)
      stage_data
      if [[ "$EXPERIMENT" == "exp7" ]]; then
        stage_pretrain
      elif [[ "$EXPERIMENT" != "exp2" && "$EXPERIMENT" != "exp4" && "$EXPERIMENT" != "exp6" ]]; then
        stage_pretrain
      fi
      if [[ "$EXPERIMENT" != "exp2" && "$EXPERIMENT" != "exp4" && "$EXPERIMENT" != "exp7" ]]; then
        stage_finetune
      fi
      stage_evaluate
      [[ "$EXPERIMENT" == "exp2" ]] && run_timing
      parse_logs
      ;;
  esac
}

main "$@"
