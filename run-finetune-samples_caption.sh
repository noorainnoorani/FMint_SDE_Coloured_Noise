export HF_HOME=/export/jyuan98/.cache/huggingface
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

num_samples=5 && seed=1 && stamp="20250811-010546" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/ornstein_uhlenbeck \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --restore_step 1000000 \
  --save_freq 500 > logs/Aug26_caption_ft_num_samples_${num_samples}/ornstein_uhlenbeck.log 2>&1

CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/double_well \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --restore_step 1000000 \
  --save_freq 500 > logs/Aug26_caption_ft_num_samples_${num_samples}/double_well.log 2>&1

CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/coupled_doublewell \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --restore_step 1000000 \
  --save_freq 500 > logs/Aug26_caption_ft_num_samples_${num_samples}/coupled_doublewell.log 2>&1

CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/mueller_overdamped \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --restore_step 1000000 \
  --save_freq 500 > logs/Aug26_caption_ft_num_samples_${num_samples}/mueller_overdamped.log 2>&1

CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/duffing_langevin \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --restore_step 1000000 \
  --save_freq 500 > logs/Aug26_caption_ft_num_samples_${num_samples}/duffing_langevin.log 2>&1

CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/perturbed_nonlinearoscillator \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --restore_step 1000000 \
  --save_freq 1000 > logs/Aug26_caption_ft_num_samples_${num_samples}/perturbed_nonlinearoscillator.log 2>&1

CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/periodic_nonlinearoscillator \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --restore_step 1000000 \
  --save_freq 1000 > logs/Aug26_caption_ft_num_samples_${num_samples}/periodic_nonlinearoscillator.log 2>&1

CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/geombrownian_motion \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --restore_step 1000000 \
  --save_freq 500 > logs/Aug26_caption_ft_num_samples_${num_samples}/geombrownian_motion.log 2>&1

CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/inhomogeneous_ornsteinuhlenbeck \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --restore_step 1000000 \
  --save_freq 500 > logs/Aug26_caption_ft_num_samples_${num_samples}/inhomogeneous_ornsteinuhlenbeck.log 2>&1

CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/stochastic_lorenz \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --restore_step 1000000 \
  --save_freq 1000 > logs/Aug26_caption_ft_num_samples_${num_samples}/stochastic_lorenz.log 2>&1

CUDA_VISIBLE_DEVICES=5 python3 run.py --problem 'icon_lm' --epochs 10 \
  --train_batch_size 5 --train_data_dirs /export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug12_${num_samples}/predator_prey \
  --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
  --steps_per_epoch 200 \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm \
  --nodeterministic --seed 1 --vistest --tfboard \
  --restore_step 1000000 \
  --save_freq 1000 > logs/Aug26_caption_ft_num_samples_${num_samples}/predator_prey.log 2>&1