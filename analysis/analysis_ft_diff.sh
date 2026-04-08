# ICON-LM
bs=10
data='duffing_langevin_noise_induced'
seed=1 && stamp="20250813-033417" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

# bs=10
# data='duffing_langevin_original'
# seed=1 && stamp="20250813-033938" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 1000 \
#  --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='duffing_langevin_overdamped'
seed=1 && stamp="20250813-034457" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='duffing_langevin_stochastic_resonance'
seed=1 && stamp="20250813-035016" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

# bs=10
# data='predator_prey_original'
# seed=1 && stamp="20250813-035539" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
#  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
#  --model_config_filename 'model_lm_config.json' \
#  --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
#  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
#  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#  --restore_step 2000 \
#  --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


bs=10
data='predator_prey_s0g4'
seed=1 && stamp="20250813-040506" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


bs=10
data='predator_prey_s2g4'
seed=1 && stamp="20250813-041433" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


bs=10
data='predator_prey_s2g6'
seed=1 && stamp="20250813-042356" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='stochastic_lorenz_chaotic' # rho = 100
seed=1 && stamp="20250813-043323" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='stochastic_lorenz_dispersion' # ρ = 28, η = 2.0
seed=1 && stamp="20250813-044245" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 1000 \
 --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

# # bs=10
# # data='stochastic_lorenz_original'
# # seed=1 && stamp="20250813-045209" && echo "seed=$seed, stamp=$stamp" &&
# # CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
# #  --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
# #  --model_config_filename 'model_lm_config.json' \
# #  --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
# #  --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
# #  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
# #  --restore_step 2000 \
# #  --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='stochastic_lorenz_spiral' #rho = 10
seed=1 && stamp="20250813-050124" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='stochastic_lorenz_rho1'
seed=1 && stamp="20250827-025514" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='stochastic_lorenz_rho13.926'
seed=1 && stamp="20250827-030439" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='stochastic_lorenz_rho20'
seed=1 && stamp="20250827-031359" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='stochastic_lorenz_rho24.5'
seed=1 && stamp="20250827-032317" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='stochastic_lorenz_rho24.06'
seed=1 && stamp="20250827-033239" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 

bs=10
data='stochastic_lorenz_rho24.76'
seed=1 && stamp="20250827-034204" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=2 python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/$data\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Aug12_ft_diff_param/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Aug12_ft_diff_param/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 


echo "Done."