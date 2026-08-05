# cuda.ml 0.4.0: GPU-accelerated machine learning from R

cuda.ml provides R interfaces to machine-learning algorithms in RAPIDS cuML.
This release updates the package for CUDA Toolkit 13.2.2, RAPIDS cuML and
nvForest 26.06, and Treelite 4.7.0. It also adds an explicit installation
workflow, broader tidymodels support, nvForest inference, and cross-process
model persistence.

The release is aimed at data scientists who work primarily in R and want to
train or run supported models on an NVIDIA GPU without managing a separate
Python environment. The package supports both direct R functions and parsnip
engines, so it can fit into an existing tidymodels workflow or be used on its
own.

## Prepare the runtime once

The R package is a portable installer and loader. After installing it, prepare
the pinned native backend and runtime for the environment:

```r
install.packages("cuda.ml")
cuda.ml::cuda_ml_install()
```

The complete runtime is currently about 1.6 GiB. The installer verifies its
artifacts and reuses the completed cache on later calls. Loading cuda.ml itself
is quiet and does not initialize CUDA.

Prebuilt backends support Linux x86_64 with glibc 2.28 or newer. GPU operations
require a supported NVIDIA GPU and driver 580 or newer. cuda.ml generally uses
one GPU for an operation; nvForest inference can target a particular GPU with
`device_id`.

## Use cuda.ml through parsnip

cuda.ml registers parsnip engines for linear and logistic regression, random
forests, nearest neighbors, and radial, polynomial, and linear support-vector
machines. For example, this fits a random-forest classifier on the GPU and
requests class probabilities through the usual parsnip interface:

```r
library(cuda.ml)
library(parsnip)

forest_spec <- rand_forest(mtry = 2, trees = 500, min_n = 5) |>
  set_mode("classification") |>
  set_engine("cuda.ml", max_depth = 20L, seed = 1L)

forest_fit <- fit(forest_spec, Species ~ ., data = iris)
predict(forest_fit, iris[1:5, ], type = "prob")
```

Portable model arguments stay in the parsnip specification. Backend-specific
controls go in `set_engine()`. Recipes can learn preprocessing on the training
data and carry it into resampling and prediction.

The direct API remains useful for clustering and dimensionality reduction,
including DBSCAN, k-means, PCA, tSVD, UMAP, and t-SNE. It also exposes
algorithm-specific features that do not have parsnip specifications, such as
the hyperbolic-tangent SVM kernel and the stochastic-gradient-descent linear
model.

## nvForest replaces FIL

The former Forest Inference Library interface has been replaced by nvForest.
nvForest runs random forests trained by `cuda_ml_rand_forest()` and can load
XGBoost models, LightGBM text models, and Treelite checkpoints. Its public API
covers prediction, model metadata, leaf identifiers, individual-tree
predictions, and checkpoint import and export.

GPU inference uses the complete runtime. A deployment that only needs nvForest
CPU inference can install a separate CUDA-free backend:

```r
cuda.ml::cuda_ml_install(device = "cpu")
```

That backend can run a cuda.ml random forest trained on a GPU as well as a
supported external tree ensemble. It does not provide cuML training or GPU
inference. The complete runtime can also run nvForest models on a CPU.

## Move fitted models between R processes

Some fitted cuda.ml objects contain native pointers that belong to one R
process. Persistence now saves explicit model state instead of relying on
`saveRDS()` to capture a live pointer.

The simplest file workflow passes a path directly. cuda.ml compresses the model
state with gzip:

```r
model <- cuda_ml_rand_forest(
  Species ~ .,
  data = iris,
  trees = 100L,
  seed = 1L
)

cuda_ml_serialize(model, "iris-forest.cuda-ml")
```

In a fresh R process, prepare the required backend, restore the fitted model,
and predict:

```r
library(cuda.ml)

model <- cuda_ml_unserialize("iris-forest.cuda-ml")

predict(model, iris[1:5, -5], type = "class")
```

Passing `connection = NULL` returns the uncompressed state as a raw vector of
bytes. That form is convenient for object stores and database BLOB columns; the
blob package can represent raw vectors for database workflows. The bundle
package is also supported for teams that already use bundled model artifacts.

Persistence is available for linear models, logistic and multinomial
regression, PCA, SVC and SVR models, UMAP, random forests, and nvForest models.
cuda.ml validates the state and required backend before restoring the model.

For nvForest-backed models, `device` on restore selects CPU or GPU inference.

## Notes for users upgrading from cuda.ml 0.3

This release updates several interfaces as part of the move to the current
upstream libraries:

- The random-forest API now uses `mtry` for predictor sampling and
  `sample_fraction` for row sampling. The unused
  `max_predictors_per_note_split` argument was removed. `trees` defaults to 100,
  and an omitted `seed` draws from R's random-number generator so `set.seed()`
  controls fitting.
- Logistic and multinomial regression now use numeric `penalty` and `mixture`
  arguments and are unregularized by default. The iteration arguments are
  `max_iter` and `linesearch_max_iter`.
- `cuda_ml_sgd()` now fits squared-loss regression only. Its `loss` argument
  was removed, and `n_iters_no_change` is now `n_iter_no_change`.
- Random projection and the KNN IVFSQ index have no replacement in the pinned
  upstream API. KNN continues to support brute-force, IVFFlat, and IVFPQ
  indexes.
- Per-call cuML logging controls were removed as a cuda.ml package policy.
- `normalize_input = TRUE` previously requested GPU-side L2 normalization. Use
  explicit preprocessing such as `recipes::step_normalize()` when appropriate;
  it centers and scales predictors and is not numerically identical to the old
  operation.
- `cuda_ml_is_classifier()` and
  `cuda_ml_can_predict_class_probabilities()` have been removed. Use
  `predict(..., type = "class")` and `predict(..., type = "prob")`; for an
  nvForest model, inspect
  `cuda_ml_nvforest_info(model)$has_probability_output` when probability support
  depends on the imported model.
- Use `cuda_ml_serialize()` and `cuda_ml_unserialize()`. The aliases with
  British spellings have been removed.

The [getting-started guide](https://mlverse.github.io/cuda.ml/articles/cuda-ml.html)
shows a first GPU workflow. The
[installation guide](https://mlverse.github.io/cuda.ml/articles/install-manage.html)
covers system requirements, caches, mirrors, audits, and source builds. See the
[tidymodels guide](https://mlverse.github.io/cuda.ml/articles/tidymodels.html)
and [model-persistence guide](https://mlverse.github.io/cuda.ml/articles/model-persistence.html)
for complete workflows.
