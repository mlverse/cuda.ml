context("Agglomerative Clustering")

test_that("cuda_ml_agglomerative_clustering() works as expected", {
  set.seed(0)

  blob_sz <- 50
  blobs <- gen_blobs(blob_sz)
  clusters <- blobs %>%
    cuda_ml_agglomerative_clustering(metric = "euclidean", n_clusters = 3L)

  expect_equal(
    clusters$labels,
    c(1L, 2L, 0L) %>%
      purrr::map(~ rep(.x, blob_sz)) %>%
      purrr::flatten_int()
  )
})
