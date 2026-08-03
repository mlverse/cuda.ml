# Load a tree ensemble with nvForest

Loads an XGBoost, LightGBM, or Treelite model with the current nvForest
API. The model's classification or regression task is read from Treelite
metadata rather than supplied separately.

## Usage

``` r
cuda_ml_nvforest_load_model(
  model_file,
  model_type = NULL,
  class_levels = NULL,
  device = c("gpu", "cpu"),
  device_id = NULL,
  layout = c("depth_first", "breadth_first", "layered"),
  precision = c("native", "single", "double"),
  default_chunk_size = NULL,
  align_bytes = NULL
)
```

## Arguments

- model_file:

  Path to a model file.

- model_type:

  File format, or `NULL` to infer it. Supported values are
  `"xgboost_ubj"`, `"xgboost_json"`, `"xgboost_legacy"`, `"lightgbm"`,
  and `"treelite_checkpoint"`.

- class_levels:

  Optional class labels in model-output order. When omitted, classifiers
  use `"0"`, `"1"`, and so on.

- device:

  Inference device: `"gpu"` or `"cpu"`.

- device_id:

  GPU device identifier, or `NULL` for the current device.

- layout:

  Tree layout.

- precision:

  Native, single, or double precision.

- default_chunk_size:

  Default prediction chunk size, or `NULL` to use nvForest's heuristic.

- align_bytes:

  Memory alignment, or `NULL` for the device default.

## Value

An nvForest model for use with
[`predict()`](https://rdrr.io/r/stats/predict.html).
