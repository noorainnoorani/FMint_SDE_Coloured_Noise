# Jan30 2026 icon SDE finetune analysis
# Results: results/Jan30_2026_ft_icon_sde/
# Dataset: /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27
# Stamps from logs: /export/jyuan98/FMint_SDE/logs/Jan30_2026_icon_ft
# No --correction: model output vs ground truth qoi_v

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

bs=10
gpu=0

# ornstein_uhlenbeck
data='ornstein_uhlenbeck'
seed=1 && stamp="20260130-222940" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan30_2026_ft_icon_sde/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Jan30_2026_ft_icon_sde/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# double_well
data='double_well'
seed=1 && stamp="20260130-223121" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan30_2026_ft_icon_sde/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Jan30_2026_ft_icon_sde/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# coupled_doublewell
data='coupled_doublewell'
seed=1 && stamp="20260130-223312" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan30_2026_ft_icon_sde/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Jan30_2026_ft_icon_sde/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# mueller_overdamped
data='mueller_overdamped'
seed=1 && stamp="20260130-223502" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan30_2026_ft_icon_sde/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Jan30_2026_ft_icon_sde/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# duffing_langevin
data='duffing_langevin'
seed=1 && stamp="20260130-223650" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan30_2026_ft_icon_sde/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Jan30_2026_ft_icon_sde/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# perturbed_nonlinearoscillator
data='perturbed_nonlinearoscillator'
seed=1 && stamp="20260130-223838" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan30_2026_ft_icon_sde/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Jan30_2026_ft_icon_sde/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# periodic_nonlinearoscillator
data='periodic_nonlinearoscillator'
seed=1 && stamp="20260130-224124" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan30_2026_ft_icon_sde/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Jan30_2026_ft_icon_sde/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# geombrownian_motion
data='geombrownian_motion'
seed=1 && stamp="20260130-224416" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan30_2026_ft_icon_sde/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Jan30_2026_ft_icon_sde/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# inhomogeneous_ornsteinuhlenbeck
data='inhomogeneous_ornsteinuhlenbeck'
seed=1 && stamp="20260130-224552" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan30_2026_ft_icon_sde/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Jan30_2026_ft_icon_sde/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# fluxgate_sensor
data='fluxgate_sensor'
seed=1 && stamp="20260130-224731" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan30_2026_ft_icon_sde/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Jan30_2026_ft_icon_sde/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# stochastic_lorenz
data='stochastic_lorenz'
seed=1 && stamp="20260130-225010" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan30_2026_ft_icon_sde/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Jan30_2026_ft_icon_sde/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# predator_prey
data='predator_prey'
seed=1 && stamp="20260130-225257" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan30_2026_ft_icon_sde/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Jan30_2026_ft_icon_sde/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

echo "Done."
