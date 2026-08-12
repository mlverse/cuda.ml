# nvForest inference and deployment

nvForest is the inference layer for random forests trained by cuda.ml
and for tree ensembles exported by XGBoost, LightGBM, or Treelite.
cuda.ml trains its random forests with cuML on a GPU. The resulting
model state is device neutral, so it can be restored for GPU inference
or deployed to a host without a GPU for CPU inference. Imported XGBoost
and LightGBM models can likewise run on either device; cuda.ml does not
train those model types.

For an externally trained XGBoost or LightGBM model, GPU inference is
the CUDA-accelerated use case. CPU mode is a deployment option that
preserves the same nvForest API on hosts without a GPU; it is not CUDA
acceleration. The GPU-training-to-CPU-deployment workflow supplied
entirely by cuda.ml applies to
[`cuda_ml_rand_forest()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_rand_forest.md)
models.

The examples in this vignette are not evaluated when the vignette is
built. Building it therefore does not require model files, a GPU, a
native backend, network access, or a runtime download.

## Prepare the runtime

Install the complete backend for cuML training or GPU inference. Install
the smaller CPU backend on a host used only for nvForest CPU inference.

``` r
library(cuda.ml)

# Complete backend for training and GPU inference.
cuda_ml_install()

# Smaller backend on a CPU-only inference host.
cuda_ml_install(device = "cpu")
```

Installation never occurs while loading or predicting. Provision the
required backend while preparing each environment. See the [installation
and runtime
guide](https://mlverse.github.io/cuda.ml/articles/install-manage.md) for
system requirements, runtime sizes, cache management, mirrors, and
source builds.

## Train on GPU and deploy on CPU

[`cuda_ml_rand_forest()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_rand_forest.md)
trains a cuML random forest on a GPU. Its fitted object uses nvForest
for inference and supports the same prediction and inspection functions
as imported models.

``` r
library(cuda.ml)
cuda_ml_install()

penguins <- palmerpenguins::penguins[
  c(
    "bill_length_mm", "bill_depth_mm", "flipper_length_mm",
    "body_mass_g", "species"
  )
]
penguins <- penguins[complete.cases(penguins), ]

forest <- cuda_ml_rand_forest(
  species ~ .,
  data = penguins,
  trees = 500,
  seed = 1
)

dir.create("penguin-forest")
cuda_ml_nvforest_export(
  forest,
  directory = "penguin-forest",
  prefix = "model"
)
```

This writes `model.treelite.checkpoint` and `model.cuda-ml.json`. Copy
both files to a CPU host. That host needs only the CPU inference
backend:

``` r
library(cuda.ml)
cuda_ml_install(device = "cpu")

penguins <- palmerpenguins::penguins[
  c(
    "bill_length_mm", "bill_depth_mm", "flipper_length_mm",
    "body_mass_g", "species"
  )
]
penguins <- penguins[complete.cases(penguins), ]

forest <- cuda_ml_nvforest_import(
  directory = "penguin-forest",
  prefix = "model",
  device = "cpu"
)

penguin_predictors <- penguins[names(penguins) != "species"]
predict(forest, penguin_predictors[1:5, ], type = "class")
predict(forest, penguin_predictors[1:5, ], type = "prob")
```

The checkpoint is a standard, device-neutral Treelite checkpoint
containing the trees. The JSON is cuda.ml metadata, not an XGBoost-style
model file. It binds the pair with the checkpoint size and SHA-256
digest and retains class labels, prediction precision, random-forest
probability semantics, R preprocessing blueprint, and processed feature
order when names are available. Exact cuda.ml round-trip requires both
files. Other Treelite consumers can load the checkpoint alone, but must
supply numeric predictors in the recorded processed order, or in the
checkpoint’s original positional order when names are absent. They must
also implement the sidecar’s labels and postprocessing semantics.

The sidecar does not encode the inference device, GPU identifier, tree
layout, chunk size, or memory alignment. Those are deployment choices.
It embeds an R-serialized preprocessing blueprint, which
[`cuda_ml_nvforest_import()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md)
unserializes. Import only artifacts from trusted sources. SHA-256 checks
integrity, not authenticity.

## Load external model formats

[`cuda_ml_nvforest_load_model()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_load_model.md)
accepts exactly five `model_type` values:

| `model_type`            | Exported model format |
|:------------------------|:----------------------|
| `"xgboost_ubj"`         | XGBoost UBJSON        |
| `"xgboost_json"`        | XGBoost JSON          |
| `"xgboost_legacy"`      | XGBoost legacy binary |
| `"lightgbm"`            | LightGBM text         |
| `"treelite_checkpoint"` | Treelite checkpoint   |

The format can always be selected explicitly:

``` r
xgb_ubjson <- cuda_ml_nvforest_load_model(
  "xgboost-model.ubj",
  model_type = "xgboost_ubj",
  device = "cpu"
)
xgb_json <- cuda_ml_nvforest_load_model(
  "xgboost-model.json",
  model_type = "xgboost_json",
  device = "cpu"
)
xgb_legacy <- cuda_ml_nvforest_load_model(
  "xgboost-model.model",
  model_type = "xgboost_legacy",
  device = "cpu"
)
lightgbm_model <- cuda_ml_nvforest_load_model(
  "lightgbm-model.txt",
  model_type = "lightgbm",
  device = "cpu"
)
treelite_model <- cuda_ml_nvforest_load_model(
  "treelite-model.checkpoint",
  model_type = "treelite_checkpoint",
  device = "cpu"
)
```

With `model_type = NULL`, the default, the loader infers four formats
from a case-insensitive file suffix:

| Suffix   | Inferred format       |
|:---------|:----------------------|
| `.ubj`   | XGBoost UBJSON        |
| `.json`  | XGBoost JSON          |
| `.model` | XGBoost legacy binary |
| `.txt`   | LightGBM text         |

``` r
xgb_model <- cuda_ml_nvforest_load_model("xgboost-model.ubj", device = "cpu")
lightgbm_model <- cuda_ml_nvforest_load_model(
  "lightgbm-model.txt",
  class_levels = c("no", "yes"),
  device = "cpu"
)
```

Treelite checkpoint filenames are not inferred. Supply
`model_type = "treelite_checkpoint"` for every checkpoint filename.

## Select CPU or GPU inference

Select the device when loading the model. The default is GPU, so set the
device explicitly for CPU deployment.

``` r
cpu_model <- cuda_ml_nvforest_load_model(
  "model.ubj",
  device = "cpu"
)

gpu_model <- cuda_ml_nvforest_load_model(
  "model.ubj",
  device = "gpu",
  device_id = 0
)
```

`device_id` applies only to GPU models. Omitting it uses the current
CUDA device. CPU and GPU models otherwise use the same prediction and
inspection APIs.

## Predict regression and classification results

Models loaded directly from XGBoost, LightGBM, or Treelite files do not
contain a cuda.ml preprocessing blueprint or feature-name mapping.
Supply numeric predictors in exactly the column order used to train and
export the model; column names are not used to reorder them.

Regression prediction returns a data frame with a `.pred` column:

``` r
regression_model <- cuda_ml_nvforest_load_model(
  "regression.ubj",
  device = "cpu"
)

new_data <- data.frame(
  feature_1 = c(0.2, 0.8),
  feature_2 = c(1.5, 0.4)
)

regression_predictions <- predict(regression_model, new_data)
```

For a classifier, `class_levels` supplies labels in model-output order.
Class prediction returns `.pred_class`. Probability prediction returns
one `.pred_<level>` column for each class.

``` r
classifier <- cuda_ml_nvforest_load_model(
  "classifier.txt",
  model_type = "lightgbm",
  class_levels = c("no", "yes"),
  device = "cpu"
)

class_predictions <- predict(classifier, new_data, type = "class")
probability_predictions <- predict(classifier, new_data, type = "prob")
```

Probability prediction is available only when the model’s Treelite
postprocessor produces probabilities.

## Serve predictions with plumber2

The following plumber2 route file restores one model when the serving
process starts and uses that model for every request. The request body
is JSON with a column-oriented `predictors` object, such as
`{"predictors":{"feature_1":[0.2],"feature_2":[1.5]}}`.

``` r
# plumber2.R
library(cuda.ml)

classifier <- cuda_ml_nvforest_import(
  directory = "models",
  prefix = "classifier",
  device = "cpu"
)

#* Return class probabilities
#*
#* @post /predict
#* @parser json
#* @serializer unboxedJSON
function(body) {
  new_data <- as.data.frame(body$predictors)
  predict(classifier, new_data, type = "prob")
}
```

Start the service with plumber2 after provisioning the CPU inference
backend:

``` r
plumber2::api("plumber2.R") |>
  plumber2::api_run(host = "0.0.0.0", port = 8000)
```

The route uses only public cuda.ml persistence and prediction APIs. Run
the installer while building or provisioning the service image so
startup and requests need no network access.

## Inspect a model

Use
[`cuda_ml_nvforest_info()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_info.md)
to inspect the task, model dimensions, output representation, inference
device, layout, precision, and chunk settings.

``` r
info <- cuda_ml_nvforest_info(classifier)
info$task_type
info$num_features
info$num_trees
info$device
info$has_probability_output
```

## Inspect leaves and individual trees

Leaf identifiers form an integer matrix with one row per observation and
one column per tree.

``` r
leaf_ids <- cuda_ml_nvforest_leaf_ids(classifier, new_data)
```

Individual-tree predictions form a numeric `rows × trees` matrix for
scalar-leaf models. Vector-leaf models return a numeric
`rows × trees × outputs` array.

``` r
per_tree <- cuda_ml_nvforest_predict_per_tree(classifier, new_data)
```

Per-tree prediction fully materializes the logical
`rows × trees × outputs` result in R; scalar-leaf models omit the
singleton output dimension. `chunk_size` changes native inference
batching, but it does not bound the R result’s memory. Account for the
complete result when choosing the number of rows and trees for this
operation.

## Persist as a single R state

Use
[`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
and
[`cuda_ml_unserialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
when a single R-native artifact is preferable. Use
[`cuda_ml_nvforest_export()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md)
and
[`cuda_ml_nvforest_import()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md)
when the Treelite checkpoint must also be available independently. Do
not save the live model object.

``` r
cuda_ml_serialize(classifier, "classifier.cuda-ml-state")
classifier <- cuda_ml_unserialize(
  "classifier.cuda-ml-state",
  device = "cpu"
)
```

Current nvForest and random-forest states are device neutral. Select the
deployment device while restoring and prepare that backend first;
cuda.ml checks it before loading the model. The [model persistence
guide](https://mlverse.github.io/cuda.ml/articles/model-persistence.md)
compares state, bundle, and checkpoint workflows.
