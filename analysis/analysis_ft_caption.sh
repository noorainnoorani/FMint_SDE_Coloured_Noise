export HF_HOME=/export/jyuan98/.cache/huggingface
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# ICON-LM
# bs=10
# data='ornstein_uhlenbeck'
# seed=1 && stamp="20250814-225532" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 1000 \
#  --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

# bs=10
# data='double_well'
# seed=1 && stamp="20250814-230708" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 1000 \
#  --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

# bs=10
# data='coupled_doublewell'
# seed=1 && stamp="20250814-231922" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 1000 \
#  --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='mueller_overdamped'
seed=1 && stamp="20250814-233134" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='duffing_langevin'
seed=1 && stamp="20250814-234352" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


bs=10
data='perturbed_nonlinearoscillator'
seed=1 && stamp="20250814-235603" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


bs=10
data='periodic_nonlinearoscillator'
seed=1 && stamp="20250815-001900" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


bs=10
data='geombrownian_motion'
seed=1 && stamp="20250815-004157" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

# bs=10
# data='inhomogeneous_ornsteinuhlenbeck'
# seed=1 && stamp="20250815-005348" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 1000 \
#  --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

# bs=10
# data='fluxgate_sensor_dt1e-4'
# seed=1 && stamp="20250815-010501" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 2000 \
#  --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='stochastic_lorenz'
seed=1 && stamp="20250815-012805" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='predator_prey'
seed=1 && stamp="20250815-015109" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

# bs=10
# data='predator_prey2'
# seed=1 && stamp="20250815-021401" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 2000 \
#  --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='fluxgate_sensor'
seed=1 && stamp="20250819-220221" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 






bs=10
data='ornstein_uhlenbeck'
seed=1 && stamp="20250822-172627" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='double_well'
seed=1 && stamp="20250822-174823" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='coupled_doublewell'
seed=1 && stamp="20250822-181119" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='inhomogeneous_ornsteinuhlenbeck'
seed=1 && stamp="20250822-183325" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug10_ft_caption/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug10_ft_caption/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


echo "Done."