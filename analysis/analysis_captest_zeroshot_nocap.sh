# ========================= feb 6th first finetune ===============================
# update the analysis.py loss_mode with caption 
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Single data folder and results folder for all runs
data_base=/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_5
results_name=Feb3_caption-sde-test-pretrain_nocap_SDE_ft_Aug12_5
mkdir -p results/$results_name

bs=10
gpu=2
data='ornstein_uhlenbeck'
seed=1 && stamp="20260207-082749" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --loss_mode nocap \
 --test_demo_num_list 1,2,3,4 \
 --test_data_dirs $data_base/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 500000 \
 --sweep_demo_nums True \
 --batch_size $bs > results/$results_name/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='double_well'
seed=1 && stamp="20260207-082749" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --loss_mode nocap \
 --test_demo_num_list 1,2,3,4 \
 --test_data_dirs $data_base/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 500000 \
 --sweep_demo_nums True \
 --batch_size $bs > results/$results_name/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='coupled_doublewell'
seed=1 && stamp="20260207-082749" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --loss_mode nocap \
 --test_demo_num_list 1,2,3,4 \
 --test_data_dirs $data_base/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 500000 \
 --sweep_demo_nums True \
 --batch_size $bs > results/$results_name/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='mueller_overdamped'
seed=1 && stamp="20260207-082749" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --loss_mode nocap \
 --test_demo_num_list 1,2,3,4 \
 --test_data_dirs $data_base/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 500000 \
 --sweep_demo_nums True \
 --batch_size $bs > results/$results_name/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='duffing_langevin'
seed=1 && stamp="20260207-082749" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --loss_mode nocap \
 --test_demo_num_list 1,2,3,4 \
 --test_data_dirs $data_base/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 500000 \
 --sweep_demo_nums True \
 --batch_size $bs > results/$results_name/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


bs=10
data='perturbed_nonlinearoscillator'
seed=1 && stamp="20260207-082749" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --loss_mode nocap \
 --test_demo_num_list 1,2,3,4 \
 --test_data_dirs $data_base/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 500000 \
 --sweep_demo_nums True \
 --batch_size $bs > results/$results_name/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


bs=10
data='periodic_nonlinearoscillator'
seed=1 && stamp="20260207-082749" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --loss_mode nocap \
 --test_demo_num_list 1,2,3,4 \
 --test_data_dirs $data_base/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 500000 \
 --sweep_demo_nums True \
 --batch_size $bs > results/$results_name/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


bs=10
data='geombrownian_motion'
seed=1 && stamp="20260207-082749" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --loss_mode nocap \
 --test_demo_num_list 1,2,3,4 \
 --test_data_dirs $data_base/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 500000 \
 --sweep_demo_nums True \
 --batch_size $bs > results/$results_name/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='inhomogeneous_ornsteinuhlenbeck'
seed=1 && stamp="20260207-082749" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --loss_mode nocap \
 --test_demo_num_list 1,2,3,4 \
 --test_data_dirs $data_base/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 500000 \
 --sweep_demo_nums True \
 --batch_size $bs > results/$results_name/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='fluxgate_sensor'
seed=1 && stamp="20260207-082749" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --loss_mode nocap \
 --test_demo_num_list 1,2,3,4 \
 --test_data_dirs $data_base/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name/icon_lm_learn_s$seed-$stamp_${data}_new \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 500000 \
 --sweep_demo_nums True \
 --batch_size $bs > results/$results_name/out_analysis_icon_lm_learn_s$seed-$stamp_${data}_new.log 2>&1 

bs=10
data='stochastic_lorenz'
seed=1 && stamp="20260207-082749" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --loss_mode nocap \
 --test_demo_num_list 1,2,3,4 \
 --test_data_dirs $data_base/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 500000 \
 --sweep_demo_nums True \
 --batch_size $bs > results/$results_name/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='predator_prey'
seed=1 && stamp="20260207-082749" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --loss_mode nocap \
 --test_demo_num_list 1,2,3,4 \
 --test_data_dirs $data_base/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 500000 \
 --sweep_demo_nums True \
 --batch_size $bs > results/$results_name/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


# ========================= additional tests: rollout_five data ===============================
# Same model, same config; data from /export/jyuan98/FMint_SDE/data_preparation/rollout_five/$data
# data_base_rollout_five=/export/jyuan98/FMint_SDE/data_preparation/rollout
# results_name_rollout_five=${results_name}_rollout
# mkdir -p results/$results_name_rollout_five

# bs=10
# data='ornstein_uhlenbeck'
# seed=1 && stamp="20260207-082749" && echo "rollout_five: seed=$seed, stamp=$stamp, data=$data" &&
# CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --loss_mode nocap \
#  --test_demo_num_list 1,2,3,4 \
#  --test_data_dirs $data_base_rollout_five/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name_rollout_five/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 500000 \
#  --sweep_demo_nums True \
#  --batch_size $bs > results/$results_name_rollout_five/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# bs=10
# data='double_well'
# seed=1 && stamp="20260207-082749" && echo "rollout_five: seed=$seed, stamp=$stamp, data=$data" &&
# CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --loss_mode nocap \
#  --test_demo_num_list 1,2,3,4 \
#  --test_data_dirs $data_base_rollout_five/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name_rollout_five/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 500000 \
#  --sweep_demo_nums True \
#  --batch_size $bs > results/$results_name_rollout_five/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# bs=10
# data='coupled_doublewell'
# seed=1 && stamp="20260207-082749" && echo "rollout_five: seed=$seed, stamp=$stamp, data=$data" &&
# CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --loss_mode nocap \
#  --test_demo_num_list 1,2,3,4 \
#  --test_data_dirs $data_base_rollout_five/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name_rollout_five/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 500000 \
#  --sweep_demo_nums True \
#  --batch_size $bs > results/$results_name_rollout_five/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# bs=10
# data='mueller_overdamped'
# seed=1 && stamp="20260207-082749" && echo "rollout_five: seed=$seed, stamp=$stamp, data=$data" &&
# CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --loss_mode nocap \
#  --test_demo_num_list 1,2,3,4 \
#  --test_data_dirs $data_base_rollout_five/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name_rollout_five/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 500000 \
#  --sweep_demo_nums True \
#  --batch_size $bs > results/$results_name_rollout_five/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# bs=10
# data='duffing_langevin'
# seed=1 && stamp="20260207-082749" && echo "rollout_five: seed=$seed, stamp=$stamp, data=$data" &&
# CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --loss_mode nocap \
#  --test_demo_num_list 1,2,3,4 \
#  --test_data_dirs $data_base_rollout_five/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name_rollout_five/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 500000 \
#  --sweep_demo_nums True \
#  --batch_size $bs > results/$results_name_rollout_five/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# bs=10
# data='perturbed_nonlinearoscillator'
# seed=1 && stamp="20260207-082749" && echo "rollout_five: seed=$seed, stamp=$stamp, data=$data" &&
# CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --loss_mode nocap \
#  --test_demo_num_list 1,2,3,4 \
#  --test_data_dirs $data_base_rollout_five/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name_rollout_five/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 500000 \
#  --sweep_demo_nums True \
#  --batch_size $bs > results/$results_name_rollout_five/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# bs=10
# data='periodic_nonlinearoscillator'
# seed=1 && stamp="20260207-082749" && echo "rollout_five: seed=$seed, stamp=$stamp, data=$data" &&
# CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --loss_mode nocap \
#  --test_demo_num_list 1,2,3,4 \
#  --test_data_dirs $data_base_rollout_five/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name_rollout_five/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 500000 \
#  --sweep_demo_nums True \
#  --batch_size $bs > results/$results_name_rollout_five/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# bs=10
# data='geombrownian_motion'
# seed=1 && stamp="20260207-082749" && echo "rollout_five: seed=$seed, stamp=$stamp, data=$data" &&
# CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --loss_mode nocap \
#  --test_demo_num_list 1,2,3,4 \
#  --test_data_dirs $data_base_rollout_five/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name_rollout_five/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 500000 \
#  --sweep_demo_nums True \
#  --batch_size $bs > results/$results_name_rollout_five/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# bs=10
# data='inhomogeneous_ornsteinuhlenbeck'
# seed=1 && stamp="20260207-082749" && echo "rollout_five: seed=$seed, stamp=$stamp, data=$data" &&
# CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --loss_mode nocap \
#  --test_demo_num_list 1,2,3,4 \
#  --test_data_dirs $data_base_rollout_five/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name_rollout_five/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 500000 \
#  --sweep_demo_nums True \
#  --batch_size $bs > results/$results_name_rollout_five/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# bs=10
# data='fluxgate_sensor'
# seed=1 && stamp="20260207-082749" && echo "rollout_five: seed=$seed, stamp=$stamp, data=$data" &&
# CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --loss_mode nocap \
#  --test_demo_num_list 1,2,3,4 \
#  --test_data_dirs $data_base_rollout_five/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name_rollout_five/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 500000 \
#  --sweep_demo_nums True \
#  --batch_size $bs > results/$results_name_rollout_five/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# bs=10
# data='stochastic_lorenz'
# seed=1 && stamp="20260207-082749" && echo "rollout_five: seed=$seed, stamp=$stamp, data=$data" &&
# CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --loss_mode nocap \
#  --test_demo_num_list 1,2,3,4 \
#  --test_data_dirs $data_base_rollout_five/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name_rollout_five/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 500000 \
#  --sweep_demo_nums True \
#  --batch_size $bs > results/$results_name_rollout_five/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# bs=10
# data='predator_prey'
# seed=1 && stamp="20260207-082749" && echo "rollout_five: seed=$seed, stamp=$stamp, data=$data" &&
# CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --loss_mode nocap \
#  --test_demo_num_list 1,2,3,4 \
#  --test_data_dirs $data_base_rollout_five/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name_rollout_five/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 500000 \
#  --sweep_demo_nums True \
#  --batch_size $bs > results/$results_name_rollout_five/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

# bs=10
# data='predator_prey2'
# seed=1 && stamp="20260207-082749" && echo "rollout_five: seed=$seed, stamp=$stamp, data=$data" &&
# CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --loss_mode nocap \
#  --test_demo_num_list 1,2,3,4 \
#  --test_data_dirs $data_base_rollout_five/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/$results_name_rollout_five/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 500000 \
#  --sweep_demo_nums True \
#  --batch_size $bs > results/$results_name_rollout_five/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1

echo "Done."