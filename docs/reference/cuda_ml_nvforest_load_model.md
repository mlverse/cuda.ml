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

  File format, or `NULL` to infer it from a recognized filename suffix.
  See **Model formats**.

- class_levels:

  Optional class labels in model-output order. When omitted, classifiers
  use `"0"`, `"1"`, and so on.

- device:

  Inference device: `"gpu"` or `"cpu"`. The default is `"gpu"`.

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

## Model formats

The supported `model_type` values are:

- `"xgboost_ubj"` for XGBoost UBJSON;

- `"xgboost_json"` for XGBoost JSON;

- `"xgboost_legacy"` for the legacy XGBoost binary format;

- `"lightgbm"` for LightGBM text models; and

- `"treelite_checkpoint"` for Treelite checkpoints.

When `model_type = NULL`, the format is inferred only from the
case-insensitive filename suffix: `.ubj`, `.json`, `.model`, and `.txt`
map to `"xgboost_ubj"`, `"xgboost_json"`, `"xgboost_legacy"`, and
`"lightgbm"`, respectively. Treelite checkpoints have no inferred suffix
and require `model_type = "treelite_checkpoint"`. Inference does not
inspect file contents; use an explicit type when the suffix does not
identify the format.

## Runtime requirements

GPU inference requires the complete, roughly 1.6 GiB runtime installed
by
[`cuda_ml_install()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_install.md)
and a supported NVIDIA GPU and driver. For CPU-only deployment, install
the separate, roughly 3 MiB backend with
`cuda_ml_install(device = "cpu")`. It does not install cuML or the
complete managed CUDA and RAPIDS runtime, and it requires neither an
NVIDIA GPU nor an NVIDIA driver. An existing complete backend
installation can also execute nvForest models on CPU; the separate
backend avoids that runtime in CPU-only environments.

## Persistence

Persist nvForest models with
[`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
and restore them with
[`cuda_ml_unserialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_unserialize.md).
Current states do not record CPU or GPU placement. Select the deployment
device when restoring, for example
`cuda_ml_unserialize(state, device = "cpu")`; GPU is the default. Tree
layout, chunk size, memory alignment, and GPU device identifier are
likewise restore-time settings. Prediction precision is retained unless
explicitly overridden. Schema 1 nvForest states require an exact
Treelite version match. The recorded package, CUDA, RAPIDS, nvForest,
and platform versions are provenance rather than compatibility gates.

To create a standard Treelite checkpoint together with the metadata
needed for a complete cuda.ml round-trip, use
[`cuda_ml_nvforest_export()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md)
and restore the pair with
[`cuda_ml_nvforest_import()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md).

## See also

[`cuda_ml_nvforest_info()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_info.md),
[`cuda_ml_nvforest_leaf_ids()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_leaf_ids.md),
[`cuda_ml_nvforest_predict_per_tree()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_predict_per_tree.md),
and
[`vignette("nvforest")`](https://mlverse.github.io/cuda.ml/articles/nvforest.md)
