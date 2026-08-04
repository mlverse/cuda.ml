# Train a random forest model

Trains a cuML random forest for classification or regression and returns
an nvForest-backed model for inference.

## Usage

``` r
cuda_ml_rand_forest(x, ...)

# Default S3 method
cuda_ml_rand_forest(x, ...)

# S3 method for class 'data.frame'
cuda_ml_rand_forest(
  x,
  y,
  mtry = NULL,
  trees = 100L,
  min_n = 2L,
  bootstrap = TRUE,
  sample_fraction = 1,
  max_depth = 16L,
  max_leaves = Inf,
  n_bins = 128L,
  min_samples_leaf = 1L,
  split_criterion = NULL,
  min_impurity_decrease = 0,
  max_batch_size = 4096L,
  n_streams = 4L,
  seed = NULL,
  ...
)

# S3 method for class 'matrix'
cuda_ml_rand_forest(
  x,
  y,
  mtry = NULL,
  trees = 100L,
  min_n = 2L,
  bootstrap = TRUE,
  sample_fraction = 1,
  max_depth = 16L,
  max_leaves = Inf,
  n_bins = 128L,
  min_samples_leaf = 1L,
  split_criterion = NULL,
  min_impurity_decrease = 0,
  max_batch_size = 4096L,
  n_streams = 4L,
  seed = NULL,
  ...
)

# S3 method for class 'formula'
cuda_ml_rand_forest(
  formula,
  data,
  mtry = NULL,
  trees = 100L,
  min_n = 2L,
  bootstrap = TRUE,
  sample_fraction = 1,
  max_depth = 16L,
  max_leaves = Inf,
  n_bins = 128L,
  min_samples_leaf = 1L,
  split_criterion = NULL,
  min_impurity_decrease = 0,
  max_batch_size = 4096L,
  n_streams = 4L,
  seed = NULL,
  ...
)

# S3 method for class 'recipe'
cuda_ml_rand_forest(
  x,
  data,
  mtry = NULL,
  trees = 100L,
  min_n = 2L,
  bootstrap = TRUE,
  sample_fraction = 1,
  max_depth = 16L,
  max_leaves = Inf,
  n_bins = 128L,
  min_samples_leaf = 1L,
  split_criterion = NULL,
  min_impurity_decrease = 0,
  max_batch_size = 4096L,
  n_streams = 4L,
  seed = NULL,
  ...
)
```

## Arguments

- x:

  Depending on the context:

  \* A \_\_data frame\_\_ of predictors. \* A \_\_matrix\_\_ of
  predictors. \* A \_\_recipe\_\_ specifying a set of preprocessing
  steps \* created from \[recipes::recipe()\]. \* A \_\_formula\_\_
  specifying the predictors and the outcome.

- ...:

  Optional arguments; currently unused.

- y:

  A numeric vector (for regression) or factor (for classification) of
  desired responses.

- mtry:

  Number of predictors sampled at each split. When `NULL`,
  classification uses the square root of the predictor count and
  regression uses all predictors.

- trees:

  Number of trees. Default: 100L.

- min_n:

  Minimum observations required to split a node. Default: 2L.

- bootstrap:

  Whether to sample observations with replacement.

- sample_fraction:

  Proportion of rows used for each tree, between 0 and 1. This is
  separate from `mtry`, which controls predictor sampling.

- max_depth:

  Maximum tree depth. Default: 16L.

- max_leaves:

  Maximum leaves per tree, or `Inf` for no limit.

- n_bins:

  Number of candidate split bins. Default: 128L.

- min_samples_leaf:

  Minimum observations in a leaf. Default: 1L.

- split_criterion:

  Split criterion, or `NULL` for the mode default. Classification
  supports `"gini"` and `"entropy"`; regression supports `"mse"`,
  `"poisson"`, `"gamma"`, and `"inverse_gaussian"`.

- min_impurity_decrease:

  Minimum impurity decrease required for a split.

- max_batch_size:

  Maximum nodes processed in one batch. Default: 4096L.

- n_streams:

  Number of CUDA streams used while fitting. Default: 4L.

- seed:

  Random seed forwarded to cuML. When `NULL`, a seed is drawn from R's
  random-number generator, so
  [`set.seed()`](https://rdrr.io/r/base/Random.html) controls the fit.

- formula:

  A formula specifying the outcome terms on the left-hand side, and the
  predictor terms on the right-hand side.

- data:

  When a \_\_recipe\_\_ or \_\_formula\_\_ is used, `data` is specified
  as a \_\_data frame\_\_ containing the predictors and (if applicable)
  the outcome.

## Value

A random forest model for use with
[`predict()`](https://rdrr.io/r/stats/predict.html).

## Deployment

Training uses cuML and requires the complete GPU backend installed by
[`cuda_ml_install()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_install.md).
Persist the fitted model with
[`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md);
the current state is device neutral. A host without a GPU can install
the separate CPU inference backend with
`cuda_ml_install(device = "cpu")` and restore the state with
`cuda_ml_unserialize(state, device = "cpu")`. The CPU backend is roughly
3 MiB installed and does not include cuML or the complete managed CUDA
and RAPIDS runtime. To create an independently usable Treelite
checkpoint and a cuda.ml JSON sidecar instead, use
[`cuda_ml_nvforest_export()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md)
and
[`cuda_ml_nvforest_import()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md).
