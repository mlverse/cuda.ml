context("t-distributed Stochastic Neighbor Embedding")

iris_input <- iris[, names(iris) != "Species"]

test_that("cuda_ml_tsne() works as expected with 'exact' method", {
  verify_iris_embedding(cuda_ml_tsne(iris_input, method = "exact"))
})

test_that("cuda_ml_tsne() works as expected with 'fft' method", {
  verify_iris_embedding(
    cuda_ml_tsne(iris_input, method = "fft", n_iter = 50000L)
  )
})
