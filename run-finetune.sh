
# finetuned on previously trained model on "all" dataset
seed=1 && stamp="20250727-181523" && echo "seed=$seed, stamp=$stamp" &&
# CUDA_VISIBLE_DEVICES=2 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/ornstein_uhlenbeck' \
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
#   --save_freq 500 > logs/Aug1_ft_first/ornstein_uhlenbeck.log 2>&1

# CUDA_VISIBLE_DEVICES=2 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/double_well' \
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
#   --save_freq 500 > logs/Aug1_ft_first/double_well.log 2>&1

# CUDA_VISIBLE_DEVICES=2 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/coupled_doublewell' \
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
#   --save_freq 500 > logs/Aug1_ft_first/coupled_doublewell.log 2>&1

# CUDA_VISIBLE_DEVICES=2 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/mueller_overdamped' \
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
#   --save_freq 500 > logs/Aug1_ft_first/mueller_overdamped.log 2>&1

# CUDA_VISIBLE_DEVICES=2 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/duffing_langevin' \
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
#   --save_freq 500 > logs/Aug1_ft_first/duffing_langevin.log 2>&1

# CUDA_VISIBLE_DEVICES=2 python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/perturbed_nonlinearoscillator' \
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
#   --save_freq 1000 > logs/Aug1_ft_first/perturbed_nonlinearoscillator.log 2>&1

# CUDA_VISIBLE_DEVICES=2 python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/periodic_nonlinearoscillator' \
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
#   --save_freq 1000 > logs/Aug1_ft_first/periodic_nonlinearoscillator.log 2>&1

# CUDA_VISIBLE_DEVICES=2 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/geombrownian_motion' \
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
#   --save_freq 500 > logs/Aug1_ft_first/geombrownian_motion.log 2>&1

# CUDA_VISIBLE_DEVICES=2 python3 run.py --problem 'icon_lm' --epochs 5 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/inhomogeneous_ornsteinuhlenbeck' \
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
#   --save_freq 500 > logs/Aug1_ft_first/inhomogeneous_ornsteinuhlenbeck.log 2>&1

CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/fluxgate_sensor' \
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
  --save_freq 1000 > logs/Aug1_ft_first/fluxgate_sensor.log 2>&1

# CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/fluxgate_sensor_dt1e-4' \
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
#   --save_freq 1000 > logs/Aug1_ft_first/fluxgate_sensor_dt1e-4.log 2>&1

# CUDA_VISIBLE_DEVICES=2 python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/stochastic_lorenz' \
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
#   --save_freq 1000 > logs/Aug1_ft_first/stochastic_lorenz.log 2>&1

# CUDA_VISIBLE_DEVICES=2 python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/predator_prey' \
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
#   --save_freq 1000 > logs/Aug1_ft_first/predator_prey.log 2>&1

# CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/predator_prey2' \
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
#   --save_freq 1000 > logs/Aug1_ft_first/predator_prey2.log 2>&1


# # CUDA_VISIBLE_DEVICES='1,2' python3 run.py --problem 'icon_lm' --epochs 10 \
# #   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDEs/ft_periodic_nonlinearoscillator_k10' \
# #   --model_config_filename 'model_lm_config.json' \
# #   --train_config_filename 'train_lm_config.json' \
# #   --test_config_filename 'test_lm_config.json' \
# #   --train_data_globs 'train*' --test_data_globs 'test*' \
# #   --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
# #   --nodeterministic --seed 1 --vistest --tfboard > logs/Jun19_train_periodic_nonlinearoscillator_k10.log 2>&1

# # finetuned on previously trained model on "all" dataset
# # seed=1 && stamp="20250508-151418" && echo "seed=$seed, stamp=$stamp" &&
# # CUDA_VISIBLE_DEVICES=2 python3 run.py --problem 'icon_lm' --epochs 5 \
# #   --train_batch_size 10 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDEs/ft_perturbed_nonlinearoscillator' \
# #   --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
# #   --steps_per_epoch 200 \
# #   --model_config_filename 'model_lm_config.json' \
# #   --train_config_filename 'train_lm_config.json' \
# #   --test_config_filename 'test_lm_config.json' \
# #   --train_data_globs 'train*' --test_data_globs 'test*' \
# #   --test_demo_num_list 1,3,5 --model icon_lm \
# #   --nodeterministic --seed 1 --vistest --tfboard \
# #   --loss_mode nocap \
# #   --restore_step 100000 \
# #   --save_freq 500 > logs/Jun15_ft_perturbed_nonlinearoscillator.log 2>&1



# # CUDA_VISIBLE_DEVICES='1,2' python3 run.py --problem 'icon_lm' --epochs 10 \
# #   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDEs/ft_perturbed_nonlinearoscillator' \
# #   --model_config_filename 'model_lm_config.json' \
# #   --train_config_filename 'train_lm_config.json' \
# #   --test_config_filename 'test_lm_config.json' \
# #   --train_data_globs 'train*' --test_data_globs 'test*' \
# #   --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
# #   --nodeterministic --seed 1 --vistest --tfboard > logs/Jun15_train_perturbed_nonlinearoscillator.log 2>&1