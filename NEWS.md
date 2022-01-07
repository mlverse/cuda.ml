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
