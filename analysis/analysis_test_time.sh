export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

bs=25
gpu=1
data='stochastic_lorenz'
seed=1 && stamp="20250804-003919" && echo "seed=$seed, stamp=$stamp" &&
CUDA_VISIBLE_DEVICES=$gpu python3 analysis.py --correction --backend jax \
 --model 'icon_lm' --test_config_filename 'test_lm_precise_config.json' \
 --model_config_filename 'model_lm_config.json' \
 --test_data_dirs /export/jyuan98/FMint_SDE/data_preparation/Oct_21_testtime\
 --analysis_dir /export/jyuan98/FMint_SDE/analysis/results/Oct_21_testtime/icon_lm_learn_s$seed-$stamp_${data} \
 --restore_dir /export/jyuan98/FMint_SDE/save/user/ckpts/icon_lm/$stamp \
 --restore_step 2000 \
 --batch_size $bs > results/Oct_21_testtime/out_analysis_icon_lm_learn_s$seed-$stamp_${data}.log 2>&1 