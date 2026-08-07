# Run the DBSCAN clustering algorithm.

Run the DBSCAN (Density-based spatial clustering of applications with
noise) clustering algorithm.

## Usage

``` r
cuda_ml_dbscan(x, min_pts, eps)
```

## Arguments

- x:

  The input matrix or data frame. Each data point should be a row and
  should consist of numeric values only.

- min_pts, eps:

  A point `p` is a core point if at least `min_pts` are within distance
  `eps` from it.

## Value

A list containing the cluster assignments of all data points. A data
point not belonging to any cluster (i.e., "noise") will have `NA` as its
cluster assignment.

## Examples

``` r
library(cuda.ml)
if (interactive() && cuda_ml_backend_info()$runtime_installed) {
  gen_pts <- function() {
    centroids <- list(c(1000, 1000), c(-1000, -1000), c(-1000, 1000))

    pts <- centroids |>
      purrr::map(\(centroid) {
        MASS::mvrnorm(10, mu = centroid, Sigma = diag(2))
      })

    rlang::exec(rbind, !!!pts)
  }

  m <- gen_pts()
  clusters <- cuda_ml_dbscan(m, min_pts = 5, eps = 3)

  print(clusters)
}
```
