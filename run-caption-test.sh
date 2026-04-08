export HF_HOME=/export/jyuan98/.cache/huggingface
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# CUDA_VISIBLE_DEVICES='0,1' python3 run.py --problem 'icon_lm' --epochs 50 \
#   --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/caption-test-folder' \
#   --model_config_filename 'model_lm_config.json' \
#   --train_config_filename 'train_lm_config.json' \
#   --test_config_filename 'test_lm_config.json' \
#   --train_data_globs 'train*' --test_data_globs 'test*' \
#   --test_demo_num_list 1,3,4 --model icon_lm \
#   --nodeterministic --seed 1 --vistest --tfboard > logs/Feb3_caption-sde-test-pretrain.log 2>&1


CUDA_VISIBLE_DEVICES='3,4' python3 run.py --problem 'icon_lm' --epochs 50 \
  --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/caption-test-folder' \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,4 --model icon_lm --loss_mode nocap \
  --nodeterministic --seed 1 --vistest --tfboard > logs/Feb3_caption-sde-test-pretrain-nocap.log 2>&1

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
  
