# cuda.ml (development version)

# cuda.ml 0.4.0

This is a breaking release for users upgrading from the cuda.ml 0.3 release
series. It requires R 4.1 or newer and targets CUDA Toolkit 13.2.2, RAPIDS cuML
and nvForest 26.06, and Treelite 4.7.0.

## Installation and runtime

- The CRAN package is now a portable R installer and loader with no compiled
  code or bundled CUDA libraries. On Linux x86_64 with glibc 2.28 or newer,
  `cuda_ml_install()` explicitly prepares the pinned native backend and its
  approximately 1.6 GiB CUDA and RAPIDS runtime. Repeated calls reuse the
  prepared cache.

- `cuda_ml_install(source = TRUE)` builds the backend on a supported host. Its
  managed mode prepares the pinned CUDA, RAPIDS, Treelite, CMake, and Ninja
  build inputs and requires GNU C++ 14 or newer. Use
  `dependencies = "host"` to supply all native build inputs. Managed builds
  detect distinct CUDA-visible architectures when possible, while
  `architectures = "native"`, `"portable"`, or an explicit CMake architecture
  list selects the build target policy.

- Loading cuda.ml is silent; it does not inspect a GPU, load native code,
  modify the cache, or use the network. `cuda_ml_backend_info()` reports pinned
  versions and cache status without those actions.
  `cuda_ml_runtime_audit()` is the separate, explicit full content and native
  registration check, and `cuda_ml_cache_clean()` removes managed cache
  generations.

- CPU nvForest inference requires a prepared cuda.ml backend. It can use the
  complete runtime installed by `cuda_ml_install()` or the separate CPU-only
  backend installed by `cuda_ml_install(device = "cpu")`. Selecting
  `device = "cpu"` when loading or restoring a model does not itself prepare a
  backend.

- Prebuilt GPU backends contain real targets for compute capabilities 7.5,
  8.0, 8.6, 8.9, 9.0, 10.0, and 12.0, plus PTX forward compatibility from
  12.0.

## Models and public APIs

- The former FIL interface was replaced by nvForest. Use
  `cuda_ml_nvforest_load_model()` for XGBoost models, LightGBM text models, and
  Treelite checkpoints, standard `predict()` for inference, and the
  `cuda_ml_nvforest_*()` inspection, checkpoint export, and import functions
  for nvForest-specific operations. Random-forest fits now use nvForest for
  prediction and persistence as well.

- The random-forest API changed. `mtry` controls predictor sampling,
  `sample_fraction` controls row sampling and defaults to 1, and the separate
  `max_predictors_per_note_split` argument was removed. When `mtry` is omitted,
  classification uses the square root of the predictor count and regression
  uses all predictors. `trees` defaults to 100. When `seed` is omitted, it is
  drawn from R's random-number generator, so `set.seed()` controls the fit. The
  `max_batch_size` and `n_streams` defaults are now 4096 and 4. Regression split
  criteria are now `"mse"`, `"poisson"`, `"gamma"`, and
  `"inverse_gaussian"`; `"mae"` was removed. Classification predictions use
  `predict(..., type = "class")` or `predict(..., type = "prob")`.

- Logistic and multinomial regression now use numeric `penalty` and `mixture`
  arguments consistent with parsnip instead of the previous `penalty`, `C`,
  and `l1_ratio` combination. The default is now unregularized; set a numeric
  `penalty` to request regularization. The iteration arguments are now
  `max_iter` and `linesearch_max_iter`; `lbfgs_memory` and
  `penalty_normalized` provide additional solver controls.
  `cuda_ml_linear_reg()` provides the corresponding parsnip-style routing
  across OLS, ridge, lasso, and elastic-net fits.

- The `normalize_input` argument was removed from OLS, ridge, lasso, and
  elastic-net models. It previously requested GPU-side L2 normalization.
  `recipes::step_normalize()` is the recommended explicit preprocessing step
  when centering and scaling are appropriate, but it is not numerically
  identical to the former L2 operation.

- `cuda_ml_sgd()` now fits squared-loss regression only, so its `loss` argument
  was removed, and `n_iters_no_change` was renamed to `n_iter_no_change`.
  Prediction methods now consistently use `new_data`; KNN classification uses
  `type = "class"` or `type = "prob"` instead of
  `output_class_probabilities`.

- Parsnip is optional. cuda.ml registers engines when parsnip is loaded,
  including `linear_reg()`, `logistic_reg()`, and `multinom_reg()` engines that
  use the usual `penalty` and `mixture` arguments.

- Random projection and KNN IVFSQ were removed and have no current replacement
  in the pinned upstream API. KNN continues to support brute-force, IVFFlat,
  and IVFPQ search. The unused `use_precomputed_tables` argument was removed
  from `cuda_ml_knn_algo_ivfpq()`.

- The per-call `cuML_log_level` arguments were removed. `has_cuML()`,
  `cuML_major_version()`, and `cuML_minor_version()` were removed in favor of
  fields returned by `cuda_ml_backend_info()`.

- `cuda_ml_is_classifier()` and
  `cuda_ml_can_predict_class_probabilities()` were removed. Use the documented
  `predict()` types for each model; for nvForest models,
  `cuda_ml_nvforest_info()` reports `task_type` and
  `has_probability_output`. The `cuda_ml_serialise()` and
  `cuda_ml_unserialise()` aliases were also removed; use
  `cuda_ml_serialize()` and `cuda_ml_unserialize()`.

## Model persistence

- `cuda_ml_serialize()` and `cuda_ml_unserialize()` now provide durable model
  states for OLS, ridge, lasso, elastic-net, SGD, logistic and multinomial
  regression, PCA, binary and one-vs-rest SVC, SVR, UMAP, random forest, and
  nvForest models. Passing a file path writes or reads a gzip-compressed state;
  open connections and in-memory raw vectors are also supported. KNN and TSVD
  fits are not currently supported: the pinned KNN API does not expose portable
  approximate-index state, and the current TSVD binding does not reconstruct
  its native transform parameters.

- `bundle::bundle()` stores the same explicit state for workflows that use the
  bundle package. Both interfaces support saving an artifact and restoring it
  in a fresh R process after the target environment prepares the required
  backend. nvForest models can also be exported and imported as a Treelite
  checkpoint plus cuda.ml metadata.

- cuda.ml validates each saved state and its required backend before restoring
  the model.

## Documentation

- The function reference is reorganized and includes guides for getting
  started, installation and runtime management, tidymodels, model persistence,
  and nvForest inference and deployment.

# cuml 0.3.2

### Linear Models

- Implemented support for logistic regression.

### Misc

- {cuda.ml} installation process will no longer require the presence of cmake
  v3.21+. If the required version of cmake is absent, then the installation
  process will use a temporary copy of cmake downloaded from
  https://github.com/Kitware/CMake/releases.

- Moving forward, Daniel Falbel (https://github.com/dfalbel) will be the new
  maintainer of {cuda.ml}.

### Linear Models

- Implemented support for logistic regression.

### Misc

- {cuda.ml} installation process will no longer require the presence of cmake
  v3.21+. If the required version of cmake is absent, then the installation
  process will use a temporary copy of cmake downloaded from
  https://github.com/Kitware/CMake/releases.

- Moving forward, Daniel Falbel (https://github.com/dfalbel) will be the new
  maintainer of {cuda.ml}.

# cuml 0.3.1

### Misc

- Addressed feedback from CRAN. Debugging symbols were previously stripped from
  a DSO to reduce package size. Now debugging symbols are preserved in
  accordance with the CRAN policy.

# cuml 0.3.0

### Linear Models

- Added support for OLS, ridge regression, and LASSO regression.

### Misc

- Fixed issue with CUDA architecture string being empty when building {cuda.ml}

- {cuda.ml} source code was revised to be compatible with `libcuml++` version
  21.06, 21.08, and 21.10

- Added support for automatically downloading a pre-built version of `libcuml++`
  and bundling & linking the downloaded `libcuml++` with the rest of the
  {cuda.ml} installation when no pre-existing copy of `libcuml++` is found. This
  is done so that new users can try out {cuda.ml} quickly without having to
  install Conda or to build `libcuml++` from source manually.

# cuml 0.2.0

### R Interface Improvements

- Re-wrote R interfaces of all supervised ML algorithms using {hardhat} to
  support data-frame, matrix, formula, and recipe inputs, per suggestion from
  @topepo in https://github.com/mlverse/cuml/issues/78 and
  https://github.com/mlverse/cuml/issues/77.

- Added {parsnip} bindings for random forest, SVM, and KNN models.

- Improved warning message for missing linkage to the RAPIDS CuML shared
  library. If the C++/CUDA source code of this package was not linked with a
  valid version of the RAPIDS CuML shared library when the package was
  installed, then a warning will be emitted whenever the package is loaded.

### Clustering

- Added support K-Means initialization options (namely, "kmeans++", "random",
  and "array") and other configuration parameters for K-Means clustering in
  `cuML`.

- Added 'cuml_log_level' option to `cuml_dbscan()`.

- Implemented R interface for single-linkage agglomerative clustering.

### Dimensionality Reduction

- Implemented R interfaces for Principal Component Analysis (PCA), Truncated
  Singular Value Decomposition (TSVD), T-distributed Stochastic Neighbor
  Embedding (T-SNE), Uniform Manifold Approximation and Projection (UMAP),
  and Random Projection routines in `cuML` (including inverse transformations
  from lower-dimensional representation to the original feature space when
  applicable).

### Nonlinear Models for Regression or Classification

- Added R interface for CuML Forest Inference Library (FIL). Users can load any
  existing XGBoost or LightGBM model using Treelite and use the model to perform
  high-throughput batch inference using GPU acceleration provided by FIL.

- Implemented R interface for K-Nearest Neighbor (KNN) classification and
  regression.

### Concurrency

- Fixed a missing `cudaEventRecord()` call in `cuml4r::async_copy()`.

### Misc

- Added `ellipsis::check_dots_used()` checks for all `...` parameters in R.

- Renamed this package from {cuml4r} to {cuml} per suggestion from
  @lorenzwalthert (context: https://github.com/mlverse/cuml/issues/75). The new
  name is shorter, and more importantly, is consistent with the mlverse naming
  convention for R packages (e.g., {keras}, {tensorflow}, {torch}, {tabnet},
  etc).

# cuml 0.1.0

### Clustering

- Implemented R interfaces for single-GPU versions of DBSCAN and K-Means
  clustering algorithms from `cuML`.

### Nonlinear Models for Regression or Classification

- Implemented R interfaces for `cuML` Random Forest classification and
  regression routines.

- Implemented R interfaces for `cuML` Support Vector Machine classifier and
  regressor.

- Support for SVM multi-class classification was implemented using the one-vs-
  rest strategy (as SVM classifier from `cuML` currently only supports binary
  classifications).

### Documentation

- Included suggestions on how to build and install `cuML` libraries from source
  with or without multi-GPU support in
  https://github.com/yitao-li/cuml-installation-notes. All suggestions are known
  to be working for RAPIDS cuML version 21.08. Please note the building-from-
  source option is more for advanced use cases that require customizations of
  RAPIDS cuML libraries' build parameters, compilers, etc, and is somewhat time-
  consuming and not as beginner-friendly as installing `cuML` directly from
  Conda.

- Found and fixed a few typos and inconsistencies.

- Some examples were simplified.

- Added documentation for `predict()` functions per suggestion from @topepo in
  https://github.com/mlverse/cuml/issues/80.

### Misc

- Configuration script was revised to work with RAPIDS cuML libraries installed
  via Conda or built from source. If RAPIDS cuML libraries could not be located
  during the configuration process, then a warning message will be emitted.

- Improved on the initial prototype of {cuml} by utilizing modern C++
  constructs from `thrust` (https://github.com/NVIDIA/thrust), making the C++
  source code of this project more readable and maintainable.

- Formatted all human-written C++ source code with clang-format and all human-
  written R source code with `styler`. Rcpp-generated C++ and R source files
  will not be formatted.

- Caching of build artifacts using `ccache` can be enabled by setting the env
  variable CUML4R_ENABLE_CCACHE (e.g., one can run `R CMD build cuml` followed
  by `CUML4R_ENABLE_CCACHE=1 R CMD INSTALL cuml_0.1.0.tar.gz` to avoid re-
  compiling the same artifacts across builds. Notice this feature is intended
  for {cuml} contributors or advanced users who need to build {cuml}
  frequently, and is not enabled by default for other users.

- Some larger cpp files were split into more granular ones for faster build
  speed (if parallel build is enabled) and also greater maintainability.
