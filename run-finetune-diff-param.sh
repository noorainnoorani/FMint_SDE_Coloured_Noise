
# finetuned on previously trained model on "all" dataset
seed=1 && stamp="20250727-181523" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/duffing_langevin_noise_induced' \
#   --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 1000000 \
#   --save_freq 500 > logs/Aug12_ft_diff_param/duffing_langevin_noise_induced.log 2>&1

# CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/duffing_langevin_original' \
#   --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 1000000 \
#   --save_freq 500 > logs/Aug12_ft_diff_param/duffing_langevin_original.log 2>&1

# CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/duffing_langevin_overdamped' \
#   --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 1000000 \
#   --save_freq 500 > logs/Aug12_ft_diff_param/duffing_langevin_overdamped.log 2>&1

# CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/duffing_langevin_stochastic_resonance' \
#   --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 1000000 \
#   --save_freq 500 > logs/Aug12_ft_diff_param/duffing_langevin_stochastic_resonance.log 2>&1

# CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/predator_prey_original' \
#   --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 1000000 \
#   --save_freq 1000 > logs/Aug12_ft_diff_param/predator_prey_original.log 2>&1

# CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/predator_prey_s0g4' \
#   --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 1000000 \
#   --save_freq 1000 > logs/Aug12_ft_diff_param/predator_prey_s0g4.log 2>&1

# CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/predator_prey_s2g4' \
#   --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 1000000 \
#   --save_freq 1000 > logs/Aug12_ft_diff_param/predator_prey_s2g4.log 2>&1

# CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/predator_prey_s2g6' \
#   --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 1000000 \
#   --save_freq 500 > logs/Aug12_ft_diff_param/predator_prey_s2g6.log 2>&1

# CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/stochastic_lorenz_chaotic' \
#   --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 1000000 \
#   --save_freq 500 > logs/Aug12_ft_diff_param/stochastic_lorenz_chaotic.log 2>&1

# CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/stochastic_lorenz_dipersion' \
#   --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 1000000 \
#   --save_freq 1000 > logs/Aug12_ft_diff_param/stochastic_lorenz_dipersion.log 2>&1

# CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/stochastic_lorenz_original' \
#   --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 1000000 \
#   --save_freq 1000 > logs/Aug12_ft_diff_param/stochastic_lorenz_original.log 2>&1

# CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/stochastic_lorenz_spiral' \
#   --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
#   --steps_per_epoch 200 \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard \
#   --loss_mode nocap \
#   --restore_step 1000000 \
#   --save_freq 1000 > logs/Aug12_ft_diff_param/stochastic_lorenz_spiral.log 2>&1

CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/stochastic_lorenz_rho1' \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --loss_mode nocap \
  --restore_step 1000000 \
  --save_freq 500 > logs/Aug12_ft_diff_param/stochastic_lorenz_rho1.log 2>&1

CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/stochastic_lorenz_rho13.926' \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --loss_mode nocap \
  --restore_step 1000000 \
  --save_freq 500 > logs/Aug12_ft_diff_param/stochastic_lorenz_rho13.926.log 2>&1

CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/stochastic_lorenz_rho20' \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --loss_mode nocap \
  --restore_step 1000000 \
  --save_freq 500 > logs/Aug12_ft_diff_param/stochastic_lorenz_rho20.log 2>&1

CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/stochastic_lorenz_rho24.5' \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --loss_mode nocap \
  --restore_step 1000000 \
  --save_freq 500 > logs/Aug12_ft_diff_param/stochastic_lorenz_rho24.5.log 2>&1

CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/stochastic_lorenz_rho24.06' \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --loss_mode nocap \
  --restore_step 1000000 \
  --save_freq 500 > logs/Aug12_ft_diff_param/stochastic_lorenz_rho24.06.log 2>&1

CUDA_VISIBLE_DEVICES=3 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_diff_param_Aug12/stochastic_lorenz_rho24.76' \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --loss_mode nocap \
  --restore_step 1000000 \
  --save_freq 500 > logs/Aug12_ft_diff_param/stochastic_lorenz_rho24.76.log 2>&1