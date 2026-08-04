# nvForest model fixtures

These small regression models were generated on 2026-08-03 with CPython
3.12.13, NumPy 2.5.1, XGBoost 2.0.3, LightGBM 4.6.0, and Treelite 4.7.0.
They use two features in their original column order and the following training
data:

```python
x = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.0],
    [-1.0, -1.0],
])
y = np.array([0.0, 1.0, 1.0, 2.0, 4.0, -2.0])
```

The XGBoost booster was trained for two rounds with
`objective="reg:squarederror"`, `max_depth=2`, `eta=1`,
`min_child_weight=0`, `lambda=0`, `alpha=0`, `base_score=0`,
`tree_method="hist"`, `nthread=1`, and `seed=20260803`. Its UBJSON, JSON, and
deprecated binary representations were produced with
`Booster.save_raw(raw_format=...)`. `treelite.checkpoint` was produced from the
same booster with `treelite.frontend.from_xgboost(booster).serialize()`.

The LightGBM booster was trained for two rounds with `objective="regression"`,
`num_leaves=4`, `max_depth=2`, `min_data_in_leaf=1`, `min_data_in_bin=1`,
`learning_rate=1`, no L1 or L2 penalty, `seed=20260803`,
`deterministic=True`, `force_col_wise=True`, and `num_threads=1`. It was written
with `Booster.save_model()`.

For the rows of `x` above, the known predictions are:

- XGBoost and Treelite checkpoint: `0.5, 0.5, 1.5, 1.5, 4, -2`
- LightGBM: `0.5, 1, 1, 1.5, 4.5, -2.5`

The committed files have these SHA-256 checksums:

```text
248f6203da507ca4ef10adcc2296c6925ffeb61f812fffa5d8fe0832e9f43e24  lightgbm.txt
a4e8b08e2e05b91171dbfca747ba91663838b131ff144569158c16e2004e8057  treelite.checkpoint
a39d87c07b5f2255ada32a6d86e7f4d91adfeb2196bdad7d7cb008168d04e220  xgboost.json
6489c3438a473414a0c37cae25f7e8906ab58602c8e32a9417e5c9fbddde62c3  xgboost.model
d90d4df57e0270b0a5d8f123a1a1d26b6e7987e3345e8c998a4590cd0cefbe3f  xgboost.ubj
```
