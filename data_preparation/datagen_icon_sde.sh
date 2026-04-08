gpu=1

dir=Jan_27_icon_sde_pretrain
testeqns=25
testquests=1
traineqns=1000
# traineqns of different parameters, trainquests of equations with same parameters, 
# nums of equations with different initial values, num_repeat of equations with same initial but different noise.


CUDA_VISIBLE_DEVICES=$gpu python3 datagen_icon_sde.py --dir $dir \
    --caption_mode train --name train --eqns $traineqns --eqn_types geombrownian_motion \
    --length 5100 --dt 0.0005 --nv_step 100 --seed 107 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_icon_sde.py --dir $dir \
    --caption_mode train --name train --eqns $traineqns --eqn_types mueller_overdamped \
    --length 5100 --dt 1e-5 --nv_step 100 --seed 109 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_icon_sde.py --dir $dir \
    --caption_mode train --name train --eqns $traineqns --eqn_types periodic_nonlinearoscillator \
    --length 510 --dt 1e-5 --nv_step 10 --seed 106 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_icon_sde.py --dir $dir \
    --caption_mode train --name train --eqns $traineqns --eqn_types stochastic_lorenz \
    --length 5100 --dt 1e-4 --nv_step 100 --seed 110 &&

CUDA_VISIBLE_DEVICES=$gpu python3 datagen_icon_sde.py --dir $dir \
    --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types geombrownian_motion \
    --length 5100 --dt 0.0005 --nv_step 100 --seed 207 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_icon_sde.py --dir $dir \
    --caption_mode test --name test --eqns $testeqns --quests $testquests \
    --eqn_types mueller_overdamped --length 5100 --dt 1e-5 --nv_step 100 --seed 203 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_icon_sde.py --dir $dir \
    --caption_mode test --name test --eqns $testeqns --quests $testquests \
    --eqn_types periodic_nonlinearoscillator --length 2000 --dt 1e-5 --nv_step 10 --seed 206 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_icon_sde.py --dir $dir \
    --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types stochastic_lorenz \
    --length 5100 --dt 1e-4 --nv_step 100 --seed 210 &&

echo "Done"
