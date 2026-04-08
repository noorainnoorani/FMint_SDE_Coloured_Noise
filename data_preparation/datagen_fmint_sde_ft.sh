gpu=5

# dir=
testeqns=1
testquests=50
traineqns=1
trainquests=50
# traineqns of different parameters, trainquests of equations with same parameters, 
# nums of equations with different initial values, num_repeat of equations with same initial but different noise.

# dir=SDE_ft_Aug12_${trainquests}/ornstein_uhlenbeck
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types ornstein_uhlenbeck \
#     --length 5000 --dt 0.001 --nv_step 100 --seed 100 &&
# dir=SDE_ft_Aug12_${trainquests}/double_well
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types double_well \
#     --length 10000 --dt 1e-5 --nv_step 100 --seed 101 &&
# dir=SDE_ft_Aug12_${trainquests}/coupled_doublewell
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types coupled_doublewell \
#     --length 10000 --dt 1e-5 --nv_step 100 --seed 102 &&
# dir=SDE_ft_Aug12_${trainquests}/mueller_overdamped
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types mueller_overdamped \
#     --length 5000 --dt 1e-5 --nv_step 100 --seed 103 &&
# dir=SDE_ft_Aug12_${trainquests}/duffing_langevin
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types duffing_langevin \
#     --length 5000 --dt 1e-4 --nv_step 100 --seed 104 &&
# dir=SDE_ft_Aug12_${trainquests}/perturbed_nonlinearoscillator
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types perturbed_nonlinearoscillator \
#     --length 20000 --dt 1e-5 --nv_step 100 --seed 102 &&
# dir=SDE_ft_Aug12_${trainquests}/periodic_nonlinearoscillator
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types periodic_nonlinearoscillator \
#     --length 2000 --dt 1e-5 --nv_step 10 --seed 102 &&
# dir=SDE_ft_Aug12_${trainquests}/geombrownian_motion
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types geombrownian_motion \
#     --length 5000 --dt 0.0005 --nv_step 100 --seed 107 &&
# dir=SDE_ft_Aug12_${trainquests}/inhomogeneous_ornsteinuhlenbeck
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types inhomogeneous_ornsteinuhlenbeck \
#     --length 20000 --dt 0.001 --nv_step 100 --seed 108 &&
# dir=SDE_ft_Aug12_${trainquests}/fluxgate_sensor
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types fluxgate_sensor \
#     --length 20000 --dt 1e-3 --nv_step 100 --seed 109 &&
# dir=SDE_ft_Aug12_${trainquests}/stochastic_lorenz
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types stochastic_lorenz \
#     --length 5000 --dt 1e-4 --nv_step 100 --seed 110 &&
# dir=SDE_ft_Aug12_${trainquests}/predator_prey
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types predator_prey \
#     --length 2000 --dt 0.005 --nv_step 10 --seed 111 &&


# dir=SDE_ft_diff_param_Aug12/stochastic_lorenz_rho24.76
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types stochastic_lorenz \
#     --length 5000 --dt 1e-4 --nv_step 100 --seed 110 &&


# dir=SDE_ft_Aug12_100/predator_prey2
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types predator_prey2 \
#     --length 2000 --dt 0.005 --nv_step 10 --seed 111 &&

# dir=SDE_ft_Aug12_100/fluxgate_sensor_dt1e-4
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types fluxgate_sensor \
#     --length 20000 --dt 1e-4 --nv_step 100 --seed 109 &&


# dir=rollout_five/mueller_overdamped
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types mueller_overdamped \
#     --length 25000 --dt 1e-5 --nv_step 100 --seed 103 &&
dir=rollout_five/stochastic_lorenz
CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
    --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
    --eqn_types stochastic_lorenz \
    --length 25000 --dt 1e-4 --nv_step 100 --seed 110 &&
# dir=rollout_five/ornstein_uhlenbeck
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types ornstein_uhlenbeck \
#     --length 25000 --dt 0.001 --nv_step 100 --seed 100 &&
# dir=rollout_five/double_well
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types double_well \
#     --length 25000 --dt 1e-5 --nv_step 100 --seed 101 &&
# dir=rollout_five/coupled_doublewell
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types coupled_doublewell \
#     --length 25000 --dt 1e-5 --nv_step 100 --seed 102 &&
# dir=rollout_five/duffing_langevin
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types duffing_langevin \
#     --length 25000 --dt 1e-4 --nv_step 100 --seed 104 &&
# dir=rollout_five/perturbed_nonlinearoscillator
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types perturbed_nonlinearoscillator \
#     --length 25000 --dt 1e-5 --nv_step 100 --seed 102 &&
# dir=rollout_five/periodic_nonlinearoscillator
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types periodic_nonlinearoscillator \
#     --length 2500 --dt 1e-5 --nv_step 10 --seed 102 &&
# dir=rollout_five/geombrownian_motion
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types geombrownian_motion \
#     --length 25000 --dt 0.0005 --nv_step 100 --seed 107 &&
# dir=rollout_five/inhomogeneous_ornsteinuhlenbeck
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types inhomogeneous_ornsteinuhlenbeck \
#     --length 25000 --dt 0.001 --nv_step 100 --seed 108 &&
# dir=rollout_five/predator_prey
# CUDA_VISIBLE_DEVICES=$gpu python3 datagen_fmint_sde_ft.py --dir $dir \
#     --caption_mode train --name train --eqns $traineqns --quests $trainquests --testquests $testquests \
#     --eqn_types predator_prey \
#     --length 2500 --dt 0.005 --nv_step 10 --seed 111 &&

echo "Done"


