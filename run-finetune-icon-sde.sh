
# finetuned on previously trained model on "all" dataset
# Data: icon-sde-ft-Jan27; logs: Jan30_2026_icon_ft
seed=1 && stamp="20260127-223823" && logdir="logs/Jan30_2026_icon_ft" && echo "seed=$seed, stamp=$stamp, logdir=$logdir" &&
data_root='/export/jyuan98/FMint_SDE/data_preparation/icon-sde-ft-Jan27' &&
mkdir -p "$logdir" &&

CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 5 \
  --train_batch_size 5 --train_data_dirs "$data_root/ornstein_uhlenbeck" \
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
  --save_freq 500 > $logdir/ornstein_uhlenbeck.log 2>&1 &&
CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 5 \
  --train_batch_size 5 --train_data_dirs "$data_root/double_well" \
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
  --save_freq 500 > $logdir/double_well.log 2>&1 &&
CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 5 \
  --train_batch_size 5 --train_data_dirs "$data_root/coupled_doublewell" \
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
  --save_freq 500 > $logdir/coupled_doublewell.log 2>&1 &&
CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 5 \
  --train_batch_size 5 --train_data_dirs "$data_root/mueller_overdamped" \
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
  --save_freq 500 > $logdir/mueller_overdamped.log 2>&1 &&
CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 5 \
  --train_batch_size 5 --train_data_dirs "$data_root/duffing_langevin" \
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
  --save_freq 500 > $logdir/duffing_langevin.log 2>&1 &&
CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs "$data_root/perturbed_nonlinearoscillator" \
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
  --save_freq 1000 > $logdir/perturbed_nonlinearoscillator.log 2>&1 &&
CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs "$data_root/periodic_nonlinearoscillator" \
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
  --save_freq 1000 > $logdir/periodic_nonlinearoscillator.log 2>&1 &&
CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 5 \
  --train_batch_size 5 --train_data_dirs "$data_root/geombrownian_motion" \
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
  --save_freq 500 > $logdir/geombrownian_motion.log 2>&1 &&
CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 5 \
  --train_batch_size 5 --train_data_dirs "$data_root/inhomogeneous_ornsteinuhlenbeck" \
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
  --save_freq 500 > $logdir/inhomogeneous_ornsteinuhlenbeck.log 2>&1 &&
CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs "$data_root/fluxgate_sensor" \
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
  --save_freq 1000 > $logdir/fluxgate_sensor.log 2>&1 &&
CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs "$data_root/stochastic_lorenz" \
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
  --save_freq 1000 > $logdir/stochastic_lorenz.log 2>&1 &&
CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs "$data_root/predator_prey" \
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
  --save_freq 1000 > $logdir/predator_prey.log 2>&1
