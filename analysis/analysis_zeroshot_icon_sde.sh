# Zeroshot analysis (pretrained checkpoint 20250727-181523, no finetune)
# Same layout as analysis_ft_icon_sde.sh: results/Jan31_2026_icon_sde_zeroshot/, dataset icon-sde-ft-Jan27, no --correction
# Stamp: 20250727-181523 for all equations; restore_step 1000000

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

bs=10
gpu=0
stamp="20260127-223823"
seed=1

# ornstein_uhlenbeck
data='ornstein_uhlenbeck' && echo "seed=$seed, stamp=$stamp, data=$data" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_icon_sde_zeroshot/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000000 \
 --batch_size $bs > results/Jan31_2026_icon_sde_zeroshot/out_analysis_zeroshot_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# double_well
data='double_well' && echo "seed=$seed, stamp=$stamp, data=$data" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_icon_sde_zeroshot/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000000 \
 --batch_size $bs > results/Jan31_2026_icon_sde_zeroshot/out_analysis_zeroshot_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# coupled_doublewell
data='coupled_doublewell' && echo "seed=$seed, stamp=$stamp, data=$data" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_icon_sde_zeroshot/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000000 \
 --batch_size $bs > results/Jan31_2026_icon_sde_zeroshot/out_analysis_zeroshot_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# mueller_overdamped
data='mueller_overdamped' && echo "seed=$seed, stamp=$stamp, data=$data" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_icon_sde_zeroshot/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000000 \
 --batch_size $bs > results/Jan31_2026_icon_sde_zeroshot/out_analysis_zeroshot_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# duffing_langevin
data='duffing_langevin' && echo "seed=$seed, stamp=$stamp, data=$data" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_icon_sde_zeroshot/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000000 \
 --batch_size $bs > results/Jan31_2026_icon_sde_zeroshot/out_analysis_zeroshot_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# perturbed_nonlinearoscillator
data='perturbed_nonlinearoscillator' && echo "seed=$seed, stamp=$stamp, data=$data" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_icon_sde_zeroshot/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000000 \
 --batch_size $bs > results/Jan31_2026_icon_sde_zeroshot/out_analysis_zeroshot_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# periodic_nonlinearoscillator
data='periodic_nonlinearoscillator' && echo "seed=$seed, stamp=$stamp, data=$data" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_icon_sde_zeroshot/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000000 \
 --batch_size $bs > results/Jan31_2026_icon_sde_zeroshot/out_analysis_zeroshot_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# geombrownian_motion
data='geombrownian_motion' && echo "seed=$seed, stamp=$stamp, data=$data" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_icon_sde_zeroshot/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000000 \
 --batch_size $bs > results/Jan31_2026_icon_sde_zeroshot/out_analysis_zeroshot_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# inhomogeneous_ornsteinuhlenbeck
data='inhomogeneous_ornsteinuhlenbeck' && echo "seed=$seed, stamp=$stamp, data=$data" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_icon_sde_zeroshot/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000000 \
 --batch_size $bs > results/Jan31_2026_icon_sde_zeroshot/out_analysis_zeroshot_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# fluxgate_sensor
data='fluxgate_sensor' && echo "seed=$seed, stamp=$stamp, data=$data" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_icon_sde_zeroshot/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000000 \
 --batch_size $bs > results/Jan31_2026_icon_sde_zeroshot/out_analysis_zeroshot_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# stochastic_lorenz
data='stochastic_lorenz' && echo "seed=$seed, stamp=$stamp, data=$data" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_icon_sde_zeroshot/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000000 \
 --batch_size $bs > results/Jan31_2026_icon_sde_zeroshot/out_analysis_zeroshot_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# predator_prey
data='predator_prey' && echo "seed=$seed, stamp=$stamp, data=$data" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27/$data \
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_icon_sde_zeroshot/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000000 \
 --batch_size $bs > results/Jan31_2026_icon_sde_zeroshot/out_analysis_zeroshot_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

echo "Done."
