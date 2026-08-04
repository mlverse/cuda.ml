# Build a KNN model.

Build a k-nearest-model for classification or regression tasks.

## Usage

``` r
cuda_ml_knn(x, ...)

# Default S3 method
cuda_ml_knn(x, ...)

# S3 method for class 'data.frame'
cuda_ml_knn(
  x,
  y,
  algo = c("brute", "ivfflat", "ivfpq"),
  metric = c("euclidean", "l2", "l1", "cityblock", "taxicab", "manhattan", "braycurtis",
    "canberra", "minkowski", "lp", "chebyshev", "linf", "jensenshannon", "cosine",
    "correlation"),
  p = 2,
  neighbors = 5L,
  ...
)

# S3 method for class 'matrix'
cuda_ml_knn(
  x,
  y,
  algo = c("brute", "ivfflat", "ivfpq"),
  metric = c("euclidean", "l2", "l1", "cityblock", "taxicab", "manhattan", "braycurtis",
    "canberra", "minkowski", "lp", "chebyshev", "linf", "jensenshannon", "cosine",
    "correlation"),
  p = 2,
  neighbors = 5L,
  ...
)

# S3 method for class 'formula'
cuda_ml_knn(
  formula,
  data,
  algo = c("brute", "ivfflat", "ivfpq"),
  metric = c("euclidean", "l2", "l1", "cityblock", "taxicab", "manhattan", "braycurtis",
    "canberra", "minkowski", "lp", "chebyshev", "linf", "jensenshannon", "cosine",
    "correlation"),
  p = 2,
  neighbors = 5L,
  ...
)

# S3 method for class 'recipe'
cuda_ml_knn(
  x,
  data,
  algo = c("brute", "ivfflat", "ivfpq"),
  metric = c("euclidean", "l2", "l1", "cityblock", "taxicab", "manhattan", "braycurtis",
    "canberra", "minkowski", "lp", "chebyshev", "linf", "jensenshannon", "cosine",
    "correlation"),
  p = 2,
  neighbors = 5L,
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

- algo:

  The query algorithm to use. Must be one of {"brute", "ivfflat",
  "ivfpq"} or a KNN algorithm specification constructed using the
  `cuda_ml_knn_algo_*` family of functions. If the algorithm is
  specified by one of the `cuda_ml_knn_algo_*` functions, then values of
  all required parameters of the algorithm will need to be specified
  explicitly. If the algorithm is specified by a character vector, then
  parameters for the algorithm are generated automatically.

  Descriptions of supported algorithms: - "brute": for brute-force, slow
  but produces exact results. - "ivfflat": for inverted file, divide the
  dataset in partitions and perform search on relevant partitions
  only. - "ivfpq": for inverted file and product quantization (vectors
  are divided into sub-vectors, and each sub-vector is encoded using
  intermediary k-means clusterings to provide partial information).
  Default: "brute".

- metric:

  Distance metric to use. Must be one of {"euclidean", "l2", "l1",
  "cityblock", "taxicab", "manhattan", "braycurtis", "canberra",
  "minkowski", "lp", "chebyshev", "linf", "jensenshannon", "cosine",
  "correlation"}. The approximate algorithms support only "euclidean",
  "l2", "cosine", and "correlation". Default: "euclidean".

- p:

  Parameter for the Minkowski metric. If p = 1, then the metric is
  equivalent to manhattan distance (l1). If p = 2, the metric is
  equivalent to euclidean distance (l2).

- neighbors:

  Number of nearest neighbors to query. Default: 5L.

- formula:

  A formula specifying the outcome terms on the left-hand side, and the
  predictor terms on the right-hand side.

- data:

  When a \_\_recipe\_\_ or \_\_formula\_\_ is used, `data` is specified
  as a \_\_data frame\_\_ containing the predictors and (if applicable)
  the outcome.

## Value

A KNN model that can be used with the 'predict' S3 generic to make
predictions on new data points. The model object contains the
following: - "knn_index": a GPU pointer to the KNN index. - "algo": enum
value of the algorithm being used for the KNN query. - "metric": enum
value of the distance metric used in KNN computations. - "p": parameter
for the Minkowski metric. - "n_samples": number of input data points. -
"n_dims": dimension of each input data point.

## Examples

``` r
library(cuda.ml)

if (interactive() && cuda_ml_backend_info()$runtime_installed) {
  library(MASS)
  library(magrittr)
  library(purrr)

  set.seed(0L)

  centers <- list(c(3, 3), c(-3, -3), c(-3, 3))

  gen_pts <- function(cluster_sz) {
    pts <- centers %>%
      map(~ mvrnorm(cluster_sz, mu = .x, Sigma = diag(2)))

    rlang::exec(rbind, !!!pts) %>% as.matrix()
  }

  gen_labels <- function(cluster_sz) {
    seq_along(centers) %>%
      sapply(function(x) rep(x, cluster_sz)) %>%
      factor()
  }

  sample_cluster_sz <- 1000
  sample_pts <- cbind(
    gen_pts(sample_cluster_sz) %>% as.data.frame(),
    label = gen_labels(sample_cluster_sz)
  )

  model <- cuda_ml_knn(
    label ~ ., sample_pts, algo = "ivfflat", metric = "euclidean"
  )

  test_cluster_sz <- 10
  test_pts <- gen_pts(test_cluster_sz) %>% as.data.frame()

  predictions <- predict(model, test_pts)
  print(predictions, n = 30)
}
```
