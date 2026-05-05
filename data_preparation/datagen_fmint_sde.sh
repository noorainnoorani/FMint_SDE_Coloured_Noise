gpu=1

dir=Oct21_testtime
testeqns=25
testquests=1
traineqns=1500
# traineqns of different parameters, trainquests of equations with same parameters, 
# nums of equations with different initial values, num_repeat of equations with same initial but different noise.


# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --eqn_types ornstein_uhlenbeck \
#     --length 5000 --dt 0.001 --nv_step 100 --seed 100 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --eqn_types double_well \
#     --length 10000 --dt 1e-5 --nv_step 100 --seed 101 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --eqn_types coupled_doublewell \
    # --length 10000 --dt 1e-5 --nv_step 100 --seed 102 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --eqn_types mueller_overdamped \
#     --length 5000 --dt 1e-5 --nv_step 100 --seed 109 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --eqn_types duffing_langevin \
#     --length 5000 --dt 1e-4 --nv_step 100 --seed 104 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --eqn_types perturbed_nonlinearoscillator \
    # --length 20000 --dt 1e-5 --nv_step 100 --seed 105 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --eqn_types periodic_nonlinearoscillator \
#     --length 500 --dt 1e-5 --nv_step 10 --seed 106 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --eqn_types geombrownian_motion \
#     --length 5000 --dt 0.0005 --nv_step 100 --seed 107 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --eqn_types inhomogeneous_ornsteinuhlenbeck \
#     --length 20000 --dt 0.001 --nv_step 100 --seed 108 &&
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
    --caption_mode train --name train --eqns $traineqns --eqn_types fluxgate_sensor \
    --length 20000 --dt 1e-3 --nv_step 100 --seed 109 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --eqn_types stochastic_lorenz \
#     --length 5000 --dt 1e-4 --nv_step 100 --seed 110 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --eqn_types predator_prey \
#     --length 2000 --dt 0.005 --nv_step 10 --seed 111 &&

# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode test --name test --eqns $testeqns --quests $testquests \
#     --eqn_types ornstein_uhlenbeck --length 5000 --dt 0.001 --nv_step 100 --seed 200 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode test --name test --eqns $testeqns --quests $testquests \
#     --eqn_types double_well --length 5000 --dt 1e-5 --nv_step 100 --seed 201 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode test --name test --eqns $testeqns --quests $testquests \
#     --eqn_types coupled_doublewell --length 10000 --dt 1e-5 --nv_step 100 --seed 202 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode test --name test --eqns $testeqns --quests $testquests \
#     --eqn_types mueller_overdamped --length 20000 --dt 1e-5 --nv_step 100 --seed 203 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode test --name test --eqns $testeqns --quests $testquests \
#     --eqn_types duffing_langevin --length 5000 --dt 1e-4 --nv_step 100 --seed 204 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode test --name test --eqns $testeqns --quests $testquests \
#     --eqn_types perturbed_nonlinearoscillator --length 20000 --dt 1e-5 --nv_step 100 --seed 205 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode test --name test --eqns $testeqns --quests $testquests \
#     --eqn_types periodic_nonlinearoscillator --length 2000 --dt 1e-5 --nv_step 10 --seed 206 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types geombrownian_motion \
#     --length 20000 --dt 0.0005 --nv_step 100 --seed 207 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types inhomogeneous_ornsteinuhlenbeck \
#     --length 20000 --dt 0.001 --nv_step 100 --seed 208 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types fluxgate_sensor \
#     --length 20000 --dt 1e-3 --nv_step 100 --seed 209 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types stochastic_lorenz \
#     --length 20000 --dt 1.08e-5 --nv_step 100 --seed 210 &&
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde.py --dir $dir \
#     --caption_mode test --name test --eqns $testeqns --quests $testquests --eqn_types predator_prey \
#     --length 2000 --dt 0.005 --nv_step 10 --seed 211 &&


echo "Done"


