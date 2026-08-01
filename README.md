
<!-- README.md is generated from README.Rmd. Please edit that file -->

# cuda.ml

<!-- badges: start -->

[![CRAN_Status_Badge](https://www.r-pkg.org/badges/version/cuda.ml)](https://cran.r-project.org/package=cuda.ml)
<a href="https://www.r-pkg.org/pkg/cuda.ml"><img src="https://cranlogs.r-pkg.org/badges/cuda.ml?color=brightgreen" style=""></a>
<!-- badges: end -->

The goal of {cuda.ml} is to provide a simple and intuitive R interface
for [RAPIDS cuML](https://github.com/rapidsai/cuml). RAPIDS cuML is a
suite of GPU-accelerated machine learning libraries powered by
[CUDA](https://en.wikipedia.org/wiki/CUDA). {cuda.ml} is under active
development, and currently implements R interfaces for the algorithms
listed below (which is a subset of [algorithms supported by RAPIDS
cuML](https://github.com/rapidsai/cuml#supported-algorithms)).

### Supported Algorithms

| Category                                              | Algorithm                                                            | Notes                                                     |
|-------------------------------------------------------|----------------------------------------------------------------------|-----------------------------------------------------------|
| **Clustering**                                        | Density-Based Spatial Clustering of Applications with Noise (DBSCAN) | Only single-GPU implementation is supported at the moment |
|                                                       | K-Means                                                              | Only single-GPU implementation is supported at the moment |
|                                                       | Single-Linkage Agglomerative Clustering                              |                                                           |
| **Dimensionality Reduction**                          | Principal Components Analysis (PCA)                                  | Only single-GPU implementation is supported at the moment |
|                                                       | Truncated Singular Value Decomposition (tSVD)                        | Only single-GPU implementation is supported at the moment |
|                                                       | Uniform Manifold Approximation and Projection (UMAP)                 | Only single-GPU implementation is supported at the moment |
|                                                       | t-Distributed Stochastic Neighbor Embedding (TSNE)                   |                                                           |
| **Linear Models for Regression or Classification**    | Linear Regression (OLS)                                              |                                                           |
|                                                       | Ridge, lasso, and elastic-net linear regression                      |                                                           |
|                                                       | Logistic and multinomial regression                                  |                                                           |
| **Nonlinear Models for Regression or Classification** | Random Forest (RF) Classification                                    | Training is single-GPU; inference uses nvForest.          |
|                                                       | Random Forest (RF) Regression                                        | Training is single-GPU; inference uses nvForest.          |
|                                                       | nvForest inference for XGBoost, LightGBM, and Treelite models        | CPU and GPU inference are supported.                      |
|                                                       | K-Nearest Neighbors (KNN) Classification                             | Brute-force, IVFFlat, and IVFPQ indexes are supported.    |
|                                                       | K-Nearest Neighbors (KNN) Regression                                 | Brute-force, IVFFlat, and IVFPQ indexes are supported.    |
|                                                       | Support Vector Machine Classifier (SVC)                              |                                                           |
|                                                       | Epsilon-Support Vector Regression (SVR)                              |                                                           |

# Examples

## Using {cuda.ml} for supervised ML tasks through {parsnip}

{cuda.ml} provides {parsnip} bindings for supervised ML algorithms such
as `linear_reg`, `logistic_reg`, `multinom_reg`, `rand_forest`,
`nearest_neighbor`, `svm_rbf`, `svm_poly`, and `svm_linear`.

Regularized models follow tidymodels conventions for `penalty` and
`mixture`. When predictors need scaling, learn and apply it explicitly
with a {recipes} step such as
`step_normalize(all_numeric_predictors())`.

The following example shows how {cuda.ml} can be used as a {parsnip}
engine to build a SVM classifier.

``` r
library(dplyr, warn.conflicts = FALSE)
library(parsnip)
library(cuda.ml)
set.seed(11235)

train_inds <- iris %>%
  mutate(ind = row_number()) %>%
  group_by(Species) %>%
  slice_sample(prop = 0.7)

train_data <- iris[train_inds$ind, ]
test_data <- iris[-train_inds$ind, ]

model <- svm_rbf(mode = "classification", rbf_sigma = 10, cost = 50) %>%
  set_engine("cuda.ml") %>%
  fit(Species ~ ., data = train_data)

preds <- predict(model, test_data)

cat("Confusion matrix:\n\n")
preds %>%
  bind_cols(test_data %>% select(Species)) %>%
  yardstick::conf_mat(truth = Species, estimate = .pred_class)
```

## Using {cuda.ml} for unsupervised ML tasks

The following example shows how {cuda.ml} can be used for unsupervised
ML tasks such as k-means clustering.

``` r
library(cuda.ml)

clustering <- cuda_ml_kmeans(
  iris[, which(names(iris) != "Species")],
  k = 3, max_iters = 100
)

# Expected outcome: there is strong correlation
# between cluster labels and `iris$Species`
print(clustering)

library(dplyr, warn.conflicts = FALSE)
tibble(cluster_id = clustering$labels, species = iris$Species) %>%
  group_by(cluster_id) %>% count(species)
```

## Using {cuda.ml} for visualizations

{cuda.ml} also features R interfaces for algorithms such as UMAP and
t-SNE, which are useful when one needs to visualize clusters of
high-dimensional data points by embedding them onto low-dimensional
manifolds (i.e., 4 dimensions or fewer).

For example, the code snippet below shows how `cuda_ml_umap()` can be
used to visualize the MNIST hand-written digits dataset, and also, the
coloring based on the true label of each sample demonstrates how well
the UMAP algorithm transforms different hand writings of the same digit
into nearby points in a 2D embedding:

``` r
library(cuda.ml)
library(ggplot2)
library(magrittr)

# load mnist
source("data-raw/load-mnist.R")
str(mnist_images)
str(mnist_labels)


# flatten each image to a 1d array, combine into a matrix with 1 row per image
flatten <- function(img) {
  dim(img) <- NULL
  img
}

flattened_mnist_images <-
  mnist_images %>% asplit(3) %>% lapply(flatten) %>% do.call(rbind, .)

# embed
embedding <- cuda_ml_umap(
  flattened_mnist_images, n_components = 2, n_neighbors = 50,
  local_connectivity = 15, repulsion_strength = 10
)

str(embedding$transformed_data)

# visualize
embedding$transformed_data %>%
  as.data.frame() %>%
  dplyr::mutate(Label = factor(mnist_labels)) %>%
  ggplot(aes(x = V1, y = V2, color = Label)) +
  geom_point(alpha = .5, size = .5) +
  labs(title = "UMAP: Uniform Manifold Approximation and Projection",
       subtitle = "Two Dimensional Embedding of MNIST")
```

From this type of visualization, we can qualitatively understand the
following about the MNIST dataset:

- The dataset can be reasonably classified into some number of
  categories.
- The right number of categories may be any where between 9 and 11.
- While there are some categories that are clearly distinguishable from
  others, there are others that have less clear boundaries with their
  neighbors.
- A small fraction of data points did not fit particularly well into any
  of the categories.
- Most data points belonging to the same digit category are clustered
  together in the UMAP output

## Installation

### R-universe binary

The R-universe binary is the supported no-compiler installation for its
current Linux target: Ubuntu 26.04 (Resolute) x86_64, including WSL2
running that distribution. Use [R-universe’s Linux binary
repository](https://docs.r-universe.dev/install/binaries.html) rather
than its source-package repository:

``` r
linux_binary_repo <- function(universe) {
  r_version <- paste(
    R.version$major,
    strsplit(R.version$minor, ".", fixed = TRUE)[[1L]][1L],
    sep = "."
  )
  sprintf(
    "https://%s.r-universe.dev/bin/linux/resolute-%s/%s/",
    universe,
    R.version$arch,
    r_version
  )
}

repos <- c(
  mlverse = linux_binary_repo("mlverse"),
  CRAN = linux_binary_repo("cran")
)
stopifnot(
  identical(unname(Sys.info()[["sysname"]]), "Linux"),
  identical(R.version$arch, "x86_64"),
  grepl(
    "/bin/linux/resolute-x86_64/[0-9]+[.][0-9]+/$",
    repos[["mlverse"]]
  )
)

install.packages(
  "cuda.ml",
  repos = repos
)
```

The binary repository path selects the prebuilt tarball. Stock Linux R
does not support `type = "binary"`, so leave `type` at its default. The
`stopifnot()` check prevents accidentally installing from the source
endpoint or an unsupported architecture.

The binary contains a precompiled {cuda.ml} backend with Treelite 4.7.0
linked statically, but not the CUDA and RAPIDS runtime libraries.
Loading the package is silent and side-effect free:

``` r
library(cuda.ml)
info <- cuda_ml_backend_info()
stopifnot(identical(info$backend, "full"))
```

`library(cuda.ml)` does not inspect the GPU, create a cache, contact the
network, or load the native backend. `cuda_ml_backend_info()` reports
the packaged backend, its exact library versions, and whether its
runtime has been installed; it does not report whether a GPU can execute
a model. The assertion also catches an unavailable binary that fell back
to the CRAN-compatible source stub.

### Runtime provisioning

Prepare the exact CUDA 13.2.2 and RAPIDS cuML and nvForest 26.06 runtime
required by the binary before fitting or predicting. The current
runtime lock downloads about 1.6 GiB, so installation can take several
minutes:

``` r
cuda.ml::cuda_ml_install()
```

`cuda_ml_install()` does not require a GPU or NVIDIA driver, and
repeated calls reuse the completed cache. It does not load the backend
or initialize CUDA. Model operations do not provision the runtime
implicitly; when the cache is absent, they report the installation
command. After provisioning, GPU-backed operations require only a
supported NVIDIA GPU and driver. nvForest CPU inference does not require
a GPU or driver.

The default cache is `tools::R_user_dir("cuda.ml", "cache")`. Set
`CUDA_ML_CACHE_DIR` to use a different location:

``` r
Sys.setenv(CUDA_ML_CACHE_DIR = "/opt/cuda-ml-cache")
cuda.ml::cuda_ml_install()
```

### Supported systems

R-universe currently publishes the managed binary for Ubuntu 26.04
(Resolute) x86_64. This includes WSL2 when its Linux distribution is
Ubuntu 26.04. Following the [RAPIDS 26.06 platform
requirements](https://docs.rapids.ai/platform-support/), the binary
requires an NVIDIA driver version 580 or newer and supports GPU compute
capabilities 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, and 12.0. The compute
capability 12.0 PTX image also provides forward compatibility for newer
GPUs supported by CUDA, following [CUDA’s forward-compatibility
model](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-compiler-driver-nvcc/index.html).

Other Linux distributions, native Windows, macOS, and Linux ARM64 are
not yet supported by the managed binary.

### CRAN and source builds

CRAN checks are network-free and install an explicit stub backend.
`cuda_ml_backend_info()` reports `backend = "stub"` for that build, and
`cuda_ml_install()` directs users to the R-universe binary. Install from
R-universe when a functional binary without local compilation is
required.

Advanced local source builds remain available. Set
`CUDA_ML_BUILD_MODE=local`, supply CUDA Toolkit 13.2.2 through
`CUDA_HOME`, and supply a prefix through `CUML_PREFIX` containing cuML
and nvForest 26.06, Treelite 4.7.0 headers, and
`lib/libtreelite_static.a`. Build Treelite as position-independent code
with its default libstdc++ ABI and with OpenMP disabled. Local builds
never download or provision Treelite. Set `CUML_CUDA_ARCHITECTURES`
explicitly to the CMake CUDA architectures to compile, and set
`CUDA_ML_CXX` to GNU C++ 14 or newer. The same compiler is used for C++
sources and nvcc host compilation. Missing inputs and other library
versions fail at configuration time.

### Development version

A bare development installation from [GitHub](https://github.com/)
produces the same CRAN-compatible stub:

``` r
# install.packages("devtools")
devtools::install_github("mlverse/cuda.ml")
```

For a functional local build, set the exact toolchain and backend inputs
described above before installing:

``` r
Sys.setenv(
  CUDA_ML_BUILD_MODE = "local",
  CUDA_HOME = "/opt/cuda-13.2.2",
  CUML_PREFIX = "/opt/cuda-ml-backend",
  CUML_CUDA_ARCHITECTURES = "75",
  CUDA_ML_CXX = "g++-14"
)
devtools::install_github("mlverse/cuda.ml")
```

## Appendix

<details>
<summary>
Inspect MNIST images
</summary>

``` r
plot_mnist(1:64)
```

</details>
