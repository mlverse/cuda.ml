context("t-distributed Stochastic Neighbor Embedding")

iris_input <- iris[, names(iris) != "Species"]

test_that("cuda_ml_tsne() works as expected with 'exact' method", {
  embedding <- cuda_ml_tsne(iris_input, method = "exact")

  set.seed(0L)
  k_clust <- kmeans(embedding, centers = 3)

  # i.e., one should be able to obtain a reasonably good clustering result
  # (as measured by the BSS/TSS ratio) within very few k-means iterations on the
  # embedding.
  expect_lte(k_clust$iter, 3)
  expect_gte(k_clust$betweenss / k_clust$totss, 0.95)
})

test_that("cuda_ml_tsne() works as expected with 'fft' method", {
  embedding <- cuda_ml_tsne(iris_input, method = "fft", n_iter = 10000L)

  set.seed(0L)
  k_clust <- kmeans(embedding, centers = 3)

  # i.e., one should be able to obtain a reasonably good clustering result
  # (as measured by the BSS/TSS ratio) within very few k-means iterations on the
  # embedding.
  expect_lte(k_clust$iter, 3)
  expect_gte(k_clust$betweenss / k_clust$totss, 0.95)
})
