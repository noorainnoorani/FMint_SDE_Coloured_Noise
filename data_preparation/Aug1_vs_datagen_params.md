# SDE_ft_Aug1 vs datagen_fmint_sde_ft.py parameters

**Question:** Is the data generated with current `datagen_fmint_sde_ft.py` (and its parameters) the same as the data in `SDE_ft_Aug1` used for Aug1_ft training?

**Answer:** **Almost the same.** Equation parameters match for 10 of 12 systems. Two systems differ:

---

## Summary

| System | Match | Notes |
|--------|--------|--------|
| ornstein_uhlenbeck | Yes | 0.2240_1.2340_0.1704 |
| double_well | Yes | 0.4328_19.6173 |
| coupled_doublewell | Yes | 0.2907_14.2205 |
| mueller_overdamped | Yes | 0.9522_1.0172_1.1702_0.8868_-0.0180_-0.0026_0.3833 |
| duffing_langevin | Yes | 0.3276_-0.9177_2.1432_0.1679_4.9787_0.0871 |
| perturbed_nonlinearoscillator | Yes | 3.0014_0.6186 |
| periodic_nonlinearoscillator | Yes | 4.6397_1.9697_0.9893 |
| geombrownian_motion | Yes | 0.0321_0.0678 |
| inhomogeneous_ornsteinuhlenbeck | Yes | 1.6848_10.0847_0.5056_0.4765 |
| stochastic_lorenz | Yes | 10.9321_35.7122_1.4535_0.4961_1.4117_1.1248 |
| **fluxgate_sensor** | **No** | See below |
| **predator_prey** | **No** | See below |

---

## Differences

### 1. fluxgate_sensor

- **SDE_ft_Aug1:** `3.6707_-0.9365_0.2921` (c, ?, epsilon)
- **datagen_fmint_sde_ft.py:** stores `c, lambda_, epsilon` → `3.6707_0.2003_0.2921` (current hardcoded: c=3.6707, lambda=0.2003, epsilon=0.2921)

So the second parameter differs: Aug1 has **-0.9365**, current script has **0.2003**. The Aug1 run was likely generated with a different `lambda_` (or a different 3-param set/order).

### 2. predator_prey

- **SDE_ft_Aug1:** `...0.2606_..._0.1688` (D and sigma3)
- **datagen_fmint_sde_ft.py:** `...0.2602_..._0.1680` (D_k=0.2602, sigma3_k=0.168)

Small numerical difference in two of the 12 parameters (D: 0.2606 vs 0.2602, sigma3: 0.1688 vs 0.1680). Likely the script was changed after Aug1 data was generated, or a different seed/version was used.

---

## How this was checked

- **Data path:** `/export/jyuan98/FMint_SDE/data_preparation/SDE_ft_Aug1/<system>/` (tfrecords).
- **Params in data:** Each tfrecord example has an `equation` string like `eqn_type_params_X_Y_nv_step_alpha`; the `X_Y_...` part was extracted and compared to the param strings produced by the current `datagen_fmint_sde_ft.py` (same format as in the script’s `all_params.append(...)`).
- **Script:** `data_preparation/check_aug1_params.py` (run with `conda run -n icon python3 check_aug1_params.py`).

---

## Conclusion

- For **all systems except fluxgate_sensor and predator_prey**, data generated with the current `datagen_fmint_sde_ft.py` (and its current hardcoded parameters) uses the **same equation parameters** as in SDE_ft_Aug1.
- **fluxgate_sensor:** Different second parameter (-0.9365 vs 0.2003); data is not the same.
- **predator_prey:** Same structure, small numeric differences in D and sigma3; data is very close but not identical.

If you need byte-for-byte identical data to Aug1 for fluxgate_sensor or predator_prey, you would need to set the script’s fluxgate second parameter to -0.9365 and predator_prey’s D and sigma3 to 0.2606 and 0.1688 (or regenerate Aug1 with the current script and accept the small predator_prey change).
