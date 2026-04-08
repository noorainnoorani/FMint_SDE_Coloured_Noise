# #  Encoder-decoder ICON (Single-Modal):
# CUDA_VISIBLE_DEVICES=0 python3 run.py --problem 'icon' --epochs 100 \
#   --train_batch_size 32 --train_data_dirs '/export/users/song362/projects/in-context-operator-networks/icon-lm/data_preparation/data' \
#   --model_config_filename 'model_icon_config.json' \
#   --train_config_filename 'train_icon_config.json' \
#   --test_config_filename 'test_icon_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon --loss_mode nocap \
#   --nodeterministic --seed 1 --vistest  --tfboard

# seed=1 && stamp="20240801-173416" && echo "seed=$seed, stamp=$stamp" &&

# CUDA_VISIBLE_DEVICES='1,2' python3 run.py --problem 'icon_lm' --epochs 100 \
#   --train_batch_size 64 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/Jul27_pretrain' \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
#   --nodeterministic --seed 1 --vistest --tfboard > logs/Jul27_pretrain.log 2>&1 

# CUDA_VISIBLE_DEVICES='1,2' python3 run.py --problem 'icon_lm' --epochs 100 \
#   --train_batch_size 64 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/Aug6_pretrain' \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
#   --nodeterministic --seed 1 --vistest --tfboard > logs/Aug6_pretrain.log 2>&1

export HF_HOME=/export/jyuan98/.cache/huggingface
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES='1,2' python3 run.py --problem 'icon_lm' --epochs 1 \
  --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/Jan_27_icon_sde_pretrain' \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
  --nodeterministic --seed 1 --vistest --tfboard > logs/Jan_27_icon_sde_pretrain.log 2>&1

# CUDA_VISIBLE_DEVICES='1,2' python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDEs/perturbed_nonlinearoscillator_dt1e-5' \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
#   --nodeterministic --seed 1 --vistest --tfboard > logs/Jun25_perturbed_nonlinearoscillator_dt1e-5.log 2>&1

# CUDA_VISIBLE_DEVICES='1,2' python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDEs/periodic_nonlinearoscillator_k10' \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
#   --nodeterministic --seed 1 --vistest --tfboard > logs/Jun25_periodic_nonlinearoscillator_k10.log 2>&1

# CUDA_VISIBLE_DEVICES='1,2' python3 run.py --problem 'icon_lm' --epochs 2 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDEs/double_well' \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
#   --nodeterministic --seed 1 --vistest --tfboard > logs/May1_double_well.log 2>&1

# CUDA_VISIBLE_DEVICES='1,2' python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDEs/mueller' \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
#   --nodeterministic --seed 1 --vistest --tfboard > logs/May1_mueller.log 2>&1

# CUDA_VISIBLE_DEVICES='1,2' python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDEs/duffing_langevin' \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
#   --nodeterministic --seed 1 --vistest --tfboard > logs/May1_duffing_langevin.log 2>&1
# originally: --loss_mode nocap \

# CUDA_VISIBLE_DEVICES='1,2' python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDEs/geombrownian_motion' \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
#   --nodeterministic --seed 1 --vistest --tfboard > logs/Jul23_geombrownian_motion.log 2>&1
# # originally: --loss_mode nocap \

# CUDA_VISIBLE_DEVICES='1,2' python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDEs/inhomogeneous_ornsteinuhlenbeck' \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
#   --nodeterministic --seed 1 --vistest --tfboard > logs/Jul23_inhomogeneous_ornsteinuhlenbeck.log 2>&1
# # originally: --loss_mode nocap \

# CUDA_VISIBLE_DEVICES='1,2' python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDEs/fluxgate_sensor' \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
#   --nodeterministic --seed 1 --vistest --tfboard > logs/Jul23_fluxgate_sensor.log 2>&1
# # originally: --loss_mode nocap \

# CUDA_VISIBLE_DEVICES='1,2' python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDEs/stochastic_lorenz' \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
#   --nodeterministic --seed 1 --vistest --tfboard > logs/Jul23_stochastic_lorenz.log 2>&1
# # originally: --loss_mode nocap \

# CUDA_VISIBLE_DEVICES='1,2' python3 run.py --problem 'icon_lm' --epochs 10 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/SDEs/predator_prey' \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
#   --nodeterministic --seed 1 --vistest --tfboard > logs/Jul23_predator_prey.log 2>&1
  
