# Run the K means clustering algorithm.

Run the K means clustering algorithm.

## Usage

``` r
cuda_ml_kmeans(
  x,
  k,
  max_iters = 300,
  tol = 0,
  init_method = c("kmeans++", "random"),
  seed = 0L
)
```

## Arguments

- x:

  The input matrix or dataframe. Each data point should be a row and
  should consist of numeric values only.

- k:

  The number of clusters.

- max_iters:

  Maximum number of iterations. Default: 300.

- tol:

  Relative tolerance with regards to inertia to declare convergence.
  Default: 0 (i.e., do not use inertia-based stopping criterion).

- init_method:

  Method for initializing the centroids. Valid methods include
  "kmeans++", "random", or a matrix of k rows, each row specifying the
  initial value of a centroid. Default: "kmeans++".

- seed:

  Seed to the random number generator. Default: 0.

## Value

A list containing the cluster assignments and the centroid of each
cluster. Each centroid will be a column within the \`centroids\` matrix.

## Examples

``` r
library(cuda.ml)

if (interactive() && cuda_ml_backend_info()$runtime_installed) {
  kclust <- cuda_ml_kmeans(
    iris[names(iris) != "Species"],
    k = 3, max_iters = 100
  )

  print(kclust)
}
```
