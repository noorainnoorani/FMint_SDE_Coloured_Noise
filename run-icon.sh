export HF_HOME=/export/jyuan98/.cache/huggingface
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES='3,4' python3 run.py --problem 'icon_lm' --epochs 100 \
  --train_batch_size 32 --train_data_dirs '/export/jyuan98/FMint_SDE/data_preparation/Jan_27_icon_sde_pretrain' \
  --model_config_filename 'model_lm_config.json' \
  --train_config_filename 'train_lm_config.json' \
  --test_config_filename 'test_lm_config.json' \
  --train_data_globs 'train*' --test_data_globs 'test*' \
  --test_demo_num_list 1,3,5 --model icon_lm --loss_mode nocap \
  --nodeterministic --seed 1 --vistest --tfboard > logs/Jan27_2026_icon_sde_pretrain.log 2>&1