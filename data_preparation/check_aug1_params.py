#!/usr/bin/env python3
"""Extract equation names (params) from SDE_ft_Aug1 tfrecords and compare with datagen_fmint_sde_ft.py defaults."""
import os
import tensorflow as tf

tf.config.set_visible_devices([], device_type='GPU')

SDE_AUG1 = "/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1"

# Current datagen_fmint_sde_ft.py hardcoded params (from script) -> expected param string in equation name
EXPECTED_PARAMS = {
    "ornstein_uhlenbeck": "0.2240_1.2340_0.1704",
    "double_well": "0.4328_19.6173",
    "coupled_doublewell": "0.2907_14.2205",
    "mueller_overdamped": "0.9522_1.0172_1.1702_0.8868_-0.0180_-0.0026_0.3833",
    "duffing_langevin": "0.3276_-0.9177_2.1432_0.1679_4.9787_0.0871",
    "perturbed_nonlinearoscillator": "3.0014_0.6186",
    "periodic_nonlinearoscillator": "4.6397_1.9697_0.9893",
    "geombrownian_motion": "0.0321_0.0678",
    "inhomogeneous_ornsteinuhlenbeck": "1.6848_10.0847_0.5056_0.4765",
    "fluxgate_sensor": "3.0000_0.2921_3.6707_0.2003",  # omega, epsilon, c, lambda
    "stochastic_lorenz": "10.9321_35.7122_1.4535_0.4961_1.4117_1.1248",
    "predator_prey": "0.3311_0.0122_0.3954_0.3577_0.4742_0.5050_0.2602_0.3184_0.4269_0.1399_0.0485_0.1680",
}

def get_equation_names_from_tfrecord(path, max_examples=5):
    names = set()
    for rec in tf.data.TFRecordDataset([path]).take(max_examples):
        ex = tf.train.Example()
        ex.ParseFromString(rec.numpy())
        eq = ex.features.feature["equation"].bytes_list.value[0].decode("utf-8")
        names.add(eq)
    return names

def main():
    subdirs = sorted([d for d in os.listdir(SDE_AUG1) if os.path.isdir(os.path.join(SDE_AUG1, d))])
    print("SDE_ft_Aug1 equation params vs datagen_fmint_sde_ft.py expected params\n")
    all_match = True
    for eqn in subdirs:
        dirpath = os.path.join(SDE_AUG1, eqn)
        files = [f for f in os.listdir(dirpath) if f.endswith(".tfrecord")]
        if not files:
            print(f"  {eqn}: no tfrecord found")
            continue
        # Use first train tfrecord
        train_files = [f for f in files if "train" in f]
        path = os.path.join(dirpath, train_files[0]) if train_files else os.path.join(dirpath, files[0])
        names = get_equation_names_from_tfrecord(path, max_examples=100)
        # equation format: eqn_type_params_X_Y_nv_step_alpha
        aug1_params = set()
        for n in names:
            if "_params_" in n and "_nv_step_" in n:
                rest = n.split("_params_", 1)[1]
                params_part = rest.split("_nv_step_")[0]
                aug1_params.add(params_part)
        aug1_str = "; ".join(sorted(aug1_params))
        exp = EXPECTED_PARAMS.get(eqn, "(not in expected dict)")
        match = exp in aug1_params or (exp == "(not in expected dict)" and aug1_params)
        if not match and exp != "(not in expected dict)":
            all_match = False
        status = "OK" if match else "DIFF"
        print(f"  {eqn}: [{status}]")
        print(f"    SDE_ft_Aug1:  {aug1_str}")
        print(f"    datagen (py): {exp}")
    print("\n" + ("All systems match." if all_match else "Some systems differ (see DIFF)."))

if __name__ == "__main__":
    main()
