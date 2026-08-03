skip_if_not(run_gpu_tests, "requires the GPU test environment")

context("t-distributed Stochastic Neighbor Embedding")

iris_input <- iris[, names(iris) != "Species"]

verify_tsne_embedding <- function(embedding) {
  expect_s3_class(embedding, "cuda_ml_tsne_model")
  expect_equal(dim(embedding), c(nrow(iris_input), 2L))
  expect_true(all(is.finite(embedding)))
  expect_gt(sum(apply(embedding, 2L, stats::sd)), 0)

  set.seed(0L)
  k_clust <- kmeans(unclass(embedding), centers = 3L, nstart = 25L)
  expect_gte(k_clust$betweenss / k_clust$totss, 0.7)
}

test_that("cuda_ml_tsne() works as expected with 'exact' method", {
  verify_tsne_embedding(
    cuda_ml_tsne(iris_input, method = "exact", seed = 0L)
  )
})

test_that("cuda_ml_tsne() works as expected with 'fft' method", {
  verify_tsne_embedding(
    cuda_ml_tsne(iris_input, method = "fft", n_iter = 5000L, seed = 0L)
  )
})
