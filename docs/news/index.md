# Changelog

## cuda.ml 0.4.0

This is a breaking release targeting one backend: CUDA Toolkit 13.2.2,
RAPIDS cuML and nvForest 26.06, and Treelite 4.6.1.

- Added an Ubuntu 26.04 x86_64 backend distributed through R-universe.
  The package contains the compiled cuda.ml backend but not the
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

- Package loading and
  [`cuda_ml_backend_info()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_backend_info.md)
  are silent and side-effect free. They do not inspect the GPU, create a
  cache, contact the network, or load the native backend. CRAN checks
  use an explicit network-free stub backend.

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

- Model persistence now requires an explicit portable representation and
  records the package, schema, model ABI, and backend identity.
  Unversioned or incompatible states fail instead of attempting
  migration. Added
  [`bundle::bundle()`](https://rstudio.github.io/bundle/reference/bundle.html)
  support for models with a portable state.

- Added GPU-less fat-binary compilation for compute capabilities 7.5,
  8.0, 8.6, 8.9, 9.0, 10.0, and 12.0, with PTX forward compatibility
  from 12.0.

- Removed native compatibility branches for historical cuML releases and
  the implicit bootstrap fallbacks for Python, pip, CMake, wheel
  layouts, CUDA architectures, and missing local toolchains.
