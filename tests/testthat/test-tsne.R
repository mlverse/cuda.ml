context("t-distributed Stochastic Neighbor Embedding")

iris_input <- iris[, names(iris) != "Species"]

verify_embedding <- function(embedding) {
  set.seed(0L)
  k_clust <- kmeans(embedding, centers = 3)

  # i.e., one should be able to obtain a reasonably good clustering result
  # (as measured by the BSS/TSS ratio) within very few k-means iterations on the
  # embedding.
  expect_lte(k_clust$iter, 3)
  expect_gte(k_clust$betweenss / k_clust$totss, 0.95)

  # Use `iris$Species` to check pairs of data points from the same species are
  # mostly in the same cluster, and those from different species are mostly in
  # different clusters in the resulting clustering.
  expect_gte(
    sklearn$metrics$adjusted_rand_score(
      labels_true = iris$Species, labels_pred = k_clust$cluster
    ),
    0.7
  )
}

test_that("cuda_ml_tsne() works as expected with 'exact' method", {
  verify_embedding(cuda_ml_tsne(iris_input, method = "exact"))
})

test_that("cuda_ml_tsne() works as expected with 'fft' method", {
  verify_embedding(cuda_ml_tsne(iris_input, method = "fft", n_iter = 50000L))
})
