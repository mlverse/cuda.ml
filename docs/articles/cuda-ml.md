# Get started with cuda.ml

cuda.ml provides R interfaces to GPU-accelerated machine learning
algorithms from RAPIDS cuML. This guide installs the managed runtime,
checks its status, fits a regression model, and computes a
principal-component representation.

The examples are not evaluated when this vignette is built. Building the
vignette therefore does not require a GPU, a native backend, network
access, or a runtime download.

## Check the system requirements

Native cuda.ml operations require Linux x86_64 with glibc 2.28 or newer.
The GPU workflows in this guide also require a supported NVIDIA GPU and
an NVIDIA driver version 580 or newer. On Windows, install and run R
inside a [compatible WSL2 Linux
distribution](https://mlverse.github.io/cuda.ml/articles/install-manage.html#windows-through-wsl2);
native Windows R is not supported. macOS, Linux ARM64, musl-based Linux
distributions, and older glibc versions are not supported.

CPU-only nvForest inference has separate requirements and does not
require an NVIDIA GPU or driver. See [nvForest inference and
deployment](https://mlverse.github.io/cuda.ml/articles/nvforest.md) for
that path.

## Install cuda.ml and its runtime

Install the R package from CRAN, then explicitly provision the complete
GPU backend:

``` r
install.packages("cuda.ml")

library(cuda.ml)
cuda_ml_install()
```

Installing the R package does not install or load CUDA, RAPIDS, or the
native cuda.ml backend.
[`cuda_ml_install()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_install.md)
downloads and verifies the backend and its locked runtime libraries. A
repeated call with the same inputs reuses the completed cache.

For cache configuration, mirrors, source builds, runtime audits, and
cleanup, see [Install and manage
cuda.ml](https://mlverse.github.io/cuda.ml/articles/install-manage.md).

## Inspect the backend

Use
[`cuda_ml_backend_info()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_backend_info.md)
to inspect the backend selected for this R installation and whether its
exact managed cache is complete:

``` r
info <- cuda_ml_backend_info()

info[c(
  "package_version",
  "platform",
  "cuda_version",
  "rapids_version",
  "minimum_driver",
  "runtime_installed"
)]
```

This check is read-only. It does not access the network, inspect an
NVIDIA GPU or driver, or load native code. In particular,
`runtime_installed = TRUE` means the expected cache is complete; it is
not a GPU-readiness check.

## Fit and predict with a direct model API

Supervised model functions accept familiar formula and data-frame
inputs. This ordinary least-squares example holds out the final seven
rows of `mtcars`, fits the model on the remaining rows, and predicts the
held-out outcomes:

``` r
train <- mtcars[1:25, ]
test <- mtcars[26:32, ]

fit <- cuda_ml_ols(
  mpg ~ .,
  data = train,
  method = "qr"
)

predictions <- predict(
  fit,
  new_data = test[names(test) != "mpg"]
)

cbind(
  actual = test$mpg,
  predicted = predictions$.pred
)
```

The fitted model retains the preprocessing blueprint learned from the
formula. [`predict()`](https://rdrr.io/r/stats/predict.html) applies
that blueprint to `new_data` before sending the resulting numeric
predictors to the backend.

## Compute a lower-dimensional representation

Unsupervised and transformation functions take observations in rows and
numeric features in columns. Because the measurements below use
different units, the example scales them before calling
[`cuda_ml_pca()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_pca.md).
The function mean-centers the scaled features, fits the principal
components, and, by default, transforms the input data:

``` r
oil_predictors <- scale(
  modeldata::oils[names(modeldata::oils) != "class"]
)

pca_fit <- cuda_ml_pca(
  oil_predictors,
  n_components = 2
)

head(pca_fit$transformed_data)
pca_fit$explained_variance_ratio
```

The rows of `transformed_data` correspond to the input rows, and its
columns are the retained components. The fitted object also contains the
component matrix, feature means, singular values, explained variance,
and explained variance ratios.

## Understand returned values

cuda.ml follows these output conventions:

- Regression predictions are data frames with a `.pred` column.
- Class predictions are data frames with a factor column named
  `.pred_class`.
- Class-probability predictions have one `.pred_<level>` column per
  outcome level when the model supports probability prediction.
- Unsupervised and transformation functions return model-specific
  objects. Consult the function reference for their named components;
  for example, PCA stores the fitted-input representation in
  `transformed_data`, while k-means stores assignments in `labels` and
  centers in `centroids`.

These prediction column names are compatible with tidymodels
conventions. Most native backend inputs are converted to numeric
matrices after formula, recipe, or data-frame preprocessing. Consult
each function’s reference page for its accepted input forms and output
components.

## Choose the direct API or parsnip

Use cuda.ml’s direct functions when you need an unsupervised or
transformation algorithm, want algorithm-specific controls, or do not
need a tidymodels workflow. Use the optional parsnip engines when
cuda.ml should participate in a tidymodels workflow with consistent
model specifications, preprocessing, resampling, or tuning. Install
parsnip separately for that interface.

See [Use cuda.ml with
tidymodels](https://mlverse.github.io/cuda.ml/articles/tidymodels.md)
for supported specifications, modes, and engine arguments.

## Continue

- [Install and manage
  cuda.ml](https://mlverse.github.io/cuda.ml/articles/install-manage.md)
  covers deployment, configuration, source builds, audits, and cleanup.
- [Use cuda.ml with
  tidymodels](https://mlverse.github.io/cuda.ml/articles/tidymodels.md)
  covers parsnip and recipe workflows.
- [Save and restore
  models](https://mlverse.github.io/cuda.ml/articles/model-persistence.md)
  compares the available persistence formats.
- [nvForest inference and
  deployment](https://mlverse.github.io/cuda.ml/articles/nvforest.md)
  covers GPU training, external tree ensembles, and CPU or GPU
  inference.
