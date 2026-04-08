export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# ========================= august 12 first finetune num_sample 5 ===============================
num_samples=5
bs=10
data='ornstein_uhlenbeck'
seed=1 && stamp="20250902-150958" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_num_samples_${num_samples}/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug12_ft_num_samples_${num_samples}/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='double_well'
seed=1 && stamp="20250902-151121" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_num_samples_${num_samples}/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug12_ft_num_samples_${num_samples}/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='coupled_doublewell'
seed=1 && stamp="20250902-151249" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_num_samples_${num_samples}/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug12_ft_num_samples_${num_samples}/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='mueller_overdamped'
seed=1 && stamp="20250902-151418" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_num_samples_${num_samples}/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug12_ft_num_samples_${num_samples}/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='duffing_langevin'
seed=1 && stamp="20250902-151547" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_num_samples_${num_samples}/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug12_ft_num_samples_${num_samples}/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


bs=10
data='perturbed_nonlinearoscillator'
seed=1 && stamp="20250902-151717" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_num_samples_${num_samples}/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug12_ft_num_samples_${num_samples}/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


bs=10
data='periodic_nonlinearoscillator'
seed=1 && stamp="20250902-151930" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_num_samples_${num_samples}/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug12_ft_num_samples_${num_samples}/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


bs=10
data='geombrownian_motion'
seed=1 && stamp="20250902-152144" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_num_samples_${num_samples}/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug12_ft_num_samples_${num_samples}/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='inhomogeneous_ornsteinuhlenbeck'
seed=1 && stamp="20250902-152307" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_num_samples_${num_samples}/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug12_ft_num_samples_${num_samples}/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='stochastic_lorenz'
seed=1 && stamp="20250902-152430" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_num_samples_${num_samples}/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug12_ft_num_samples_${num_samples}/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='predator_prey'
seed=1 && stamp="20250902-152642" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_num_samples_${num_samples}/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug12_ft_num_samples_${num_samples}/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


echo "Done."