# ========================= Jan 31 finetune demo test ===============================
# Results: results/Jan31_2026_ft_demo/ ; --sweep_demo_nums (errors for 1–4 demos)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

bs=10
gpu=6
data='ornstein_uhlenbeck'
seed=1 && stamp="20250803-231238" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_ft_demo/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --sweep_demo_nums \
 --batch_size $bs > results/Jan31_2026_ft_demo/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='double_well'
seed=1 && stamp="20250803-231857" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_ft_demo/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --sweep_demo_nums \
 --batch_size $bs > results/Jan31_2026_ft_demo/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='coupled_doublewell'
seed=1 && stamp="20250803-232609" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_ft_demo/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --sweep_demo_nums \
 --batch_size $bs > results/Jan31_2026_ft_demo/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='mueller_overdamped'
seed=1 && stamp="20250803-233318" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_ft_demo/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --sweep_demo_nums \
 --batch_size $bs > results/Jan31_2026_ft_demo/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='duffing_langevin'
seed=1 && stamp="20250803-234029" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_ft_demo/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --sweep_demo_nums \
 --batch_size $bs > results/Jan31_2026_ft_demo/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


bs=10
data='perturbed_nonlinearoscillator'
seed=1 && stamp="20250803-234739" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_ft_demo/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --sweep_demo_nums \
 --batch_size $bs > results/Jan31_2026_ft_demo/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


bs=10
data='periodic_nonlinearoscillator'
seed=1 && stamp="20250803-235959" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_ft_demo/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --sweep_demo_nums \
 --batch_size $bs > results/Jan31_2026_ft_demo/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


bs=10
data='geombrownian_motion'
seed=1 && stamp="20250804-181758" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_ft_demo/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --sweep_demo_nums \
 --batch_size $bs > results/Jan31_2026_ft_demo/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='inhomogeneous_ornsteinuhlenbeck'
seed=1 && stamp="20250804-002003" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_ft_demo/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --sweep_demo_nums \
 --batch_size $bs > results/Jan31_2026_ft_demo/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='fluxgate_sensor'
seed=1 && stamp="20250819-212419" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_ft_demo/icon_lm_learn_s$seed-$stamp_${data}_new \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --sweep_demo_nums \
 --batch_size $bs > results/Jan31_2026_ft_demo/out_analysis_icon_lm_learn_s$seed-$stamp_${data}_new.log 2>&1 

bs=10
data='stochastic_lorenz'
seed=1 && stamp="20250804-003919" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_ft_demo/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --sweep_demo_nums \
 --batch_size $bs > results/Jan31_2026_ft_demo/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='predator_prey'
seed=1 && stamp="20250804-005132" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Jan31_2026_ft_demo/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --sweep_demo_nums \
 --batch_size $bs > results/Jan31_2026_ft_demo/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


# bs=10
# data='mueller_overdamped'
# seed=1 && stamp="20250803-233318" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/rollout/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/rollout/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 1000 \
#  --batch_size $bs > results/rollout/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


echo "Done."