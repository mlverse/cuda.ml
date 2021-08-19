# cuml4r 0.1.0.9000

### Clustering

- Added support K-Means initialization options (namely, "kmeans++", "random",
  and "array") and other configuration parameters for K-Means clustering in
  `cuML`.

- Added 'cuml_log_level' option to `cuml_dbscan()`.

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

### Concurrency

- Fixed a missing `cudaEventRecord()` call in `cuml4r::async_copy()`.

### Misc

- Added `ellipsis::check_dots_used()` checks for all `...` parameters in R.

# cuml4r 0.1.0

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
  to be working for `cuML` version 21.08. Please note the building-from-source
  option is more for advanced use cases that require customizations of `cuML`
  libraries' build parameters, compilers, etc, and is somewhat time-consuming
  and not as beginner-friendly as installing `cuML` directly from conda.

- Found and fixed a few typos and inconsistencies.

- Some examples were simplified.

### Misc

- Configuration script was revised to work with `cuML` libraries installed via
  conda or built from source. If `cuML` libraries could not be located during
  the configuration process, then a warning message will be emitted.

- Improved on the initial prototype of `cuml4r` by utilizing modern C++
  constructs from `thrust` (https://github.com/NVIDIA/thrust), making the C++
  source code of this project more readable and maintainable.

- Formatted all human-written C++ source code with clang-format and all human-
  written R source code with `styler`. Rcpp-generated C++ and R source files
  will not be formatted.

- Caching of build artifacts using `ccache` can be enabled by setting the env
  variable CUML4R_ENABLE_CCACHE (e.g., one can run `R CMD build cuml4r` followed
  by `CUML4R_ENABLE_CCACHE=1 R CMD INSTALL cuml4r_0.1.0.tar.gz` to avoid re-
  compiling the same artifacts across builds. Notice this feature is intended
  for `cuml4r` contributors or advanced users who need to build `cuml4r`
  frequently, and is not enabled by default for other users.

- Some larger cpp files were split into more granular ones for faster build
  speed (if parallel build is enabled) and also greater maintainability.
