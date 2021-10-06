context("Density-Based Spatial Clustering of Applications with Noise")

test_that("cuda_ml_dbscan() works as expected", {
  set.seed(0)

  blob_sz <- 10
  blobs <- gen_blobs(blob_sz)
  clusters <- cuda_ml_dbscan(blobs, min_pts = 5, eps = 3)

  expect_equal(
    clusters$labels,
    seq(0, 2) %>%
      purrr::map(~ rep(.x, blob_sz)) %>%
      purrr::flatten_int()
  )
})
