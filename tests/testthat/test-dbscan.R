context("Density-Based Spatial Clustering of Applications with Noise")

gen_pts <- function() {
  centroids <- list(c(1000, 1000), c(-1000, -1000), c(-1000, 1000))
  pts <- centroids %>%
    purrr::map(~ MASS::mvrnorm(10, mu = .x, Sigma = diag(2)))

  rlang::exec(rbind, !!!pts)
}

test_that("cuda_ml_dbscan() works as expected", {
  set.seed(0)
  pts <- gen_pts()
  clusters <- cuda_ml_dbscan(pts, min_pts = 5, eps = 3)

  expect_equal(
    clusters$labels,
    seq(0, 2) %>%
      purrr::map(~ rep(.x, 10)) %>%
      purrr::flatten_int()
  )
})
