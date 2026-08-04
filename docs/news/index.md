# Changelog

## cuda.ml 0.4.0

This is a breaking release targeting one backend: CUDA Toolkit 13.2.2,
RAPIDS cuML and nvForest 26.06, and Treelite 4.7.0.

- Added Linux x86_64 backends targeting glibc 2.28 or newer and hosted
  as GitHub Release assets. The CRAN package remains a portable R
  installer and loader; it contains neither compiled code nor the
  approximately 1.6 GiB CUDA and RAPIDS runtime.

- Added explicit
  [`cuda_ml_install()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_install.md)
  runtime preparation. Model operations no longer download libraries
  implicitly; they direct users to run the installer when the managed
  runtime is absent.
  [`cuda_ml_runtime_audit()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_runtime_audit.md)
  performs a full content audit, and
  [`cuda_ml_cache_clean()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_cache_clean.md)
  removes cuda.ml cache generations.

- Added `cuda_ml_install(source = TRUE)` for host builds without Docker
  or a prebuilt cuda.ml backend. By default it bootstraps exact locked
  CUDA, RAPIDS, Treelite, CMake, and Ninja build inputs, requiring only
  GNU C++ 14 or newer on a supported Linux host. `dependencies = "host"`
  uses explicit native build inputs and makes no downloads. Managed
  builds detect CUDA-visible GPU architectures by default and otherwise
  use the portable package list. `architectures = "native"` requires
  detection, while `architectures = "portable"` forces the package list.
  Source backend selection persists across R sessions.

- Package loading and
  [`cuda_ml_backend_info()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_backend_info.md)
  are silent and side-effect free. They do not inspect the GPU, create a
  cache, contact the network, or load the native backend. CRAN checks
  use an explicit network-free stub backend.

- Parsnip is now optional. Loading cuda.ml does not load parsnip,
  ggplot2, or S7; cuda.ml registers its engines when parsnip is loaded.

- Replaced the removed cuML FIL interface with current nvForest loading,
  prediction, model inspection, leaf-ID, per-tree, and persistence APIs.
  Random-forest inference and persistence now use the same nvForest
  backend.

- Corrected random-forest arguments: `mtry` now controls sampled
  predictors, `sample_fraction` controls sampled rows, `trees` defaults
  to 100, and `seed` is forwarded to cuML. Classification uses
  `predict(..., type = "class")` and `predict(..., type = "prob")`.

- Added parsnip engines for
  [`linear_reg()`](https://parsnip.tidymodels.org/reference/linear_reg.html),
  [`logistic_reg()`](https://parsnip.tidymodels.org/reference/logistic_reg.html),
  and
  [`multinom_reg()`](https://parsnip.tidymodels.org/reference/multinom_reg.html)
  using the customary `penalty` and `mixture` arguments. Normalization
  is no longer emulated inside regularized linear models; use a recipe
  preprocessing step when scaling is required.

- Removed interfaces that the supported backend cannot implement: FIL,
  random projection, KNN IVFSQ, cuML log-level controls, `has_cuML()`,
  and the split cuML version queries.
  [`cuda_ml_backend_info()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_backend_info.md)
  is the single backend metadata interface.

- Model persistence now stores explicit versioned state through
  [`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
  and
  [`bundle::bundle()`](https://rstudio.github.io/bundle/reference/bundle.html).
  The package version is recorded as provenance and does not by itself
  prevent restoration. Schema 1 compatibility is defined by the model
  ABI: linear and logistic-regression states require no backend identity
  match; PCA, SVC, one-vs-rest SVC, SVR, and UMAP states require the
  recorded RAPIDS version; and random-forest and nvForest states require
  the recorded Treelite version. Unknown schemas or ABIs, missing
  payloads, and missing or incompatible required backend fields fail
  without implicit migration or native-pointer fallback behavior.

- Added GPU-less fat-binary compilation for compute capabilities 7.5,
  8.0, 8.6, 8.9, 9.0, 10.0, and 12.0, with PTX forward compatibility
  from 12.0.

- Removed native compatibility branches for historical cuML releases and
  the implicit bootstrap fallbacks for Python, pip, CMake, wheel
  layouts, CUDA architectures, and missing local toolchains.
