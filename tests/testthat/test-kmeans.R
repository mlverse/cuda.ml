context("K-Means")

sklearn_kmeans_model <- sklearn$cluster$KMeans(
  n_clusters = 3L, max_iter = 100L
)
sklearn_kclust <- sklearn_kmeans_model$fit(sklearn_iris_dataset$data)

verify_cluster_centers <- function(centers) {
  expect_equal(
    sort_mat(centers),
    sort_mat(sklearn_kclust$cluster_centers_),
    tolerance = 0.01,
    scale = 1
  )
}

test_that("cuda_ml_kmeans() works as expected with 'kmeans++' initialization method", {
  cuda_ml_kclust <- cuda_ml_kmeans(
    iris[, which(names(iris) != "Species")],
    k = 3,
    max_iters = 100,
    init_method = "kmeans++"
  )

  verify_cluster_centers(cuda_ml_kclust$centroids)
})

test_that("cuda_ml_kmeans() works as expected with 'random' initialization method", {
  cuda_ml_kclust <- cuda_ml_kmeans(
    iris[, which(names(iris) != "Species")],
    k = 3,
    max_iters = 100,
    init_method = "random"
  )

  expect_equal(dim(cuda_ml_kclust$centroids), c(3L, 4L))
  expect_equal(length(cuda_ml_kclust$labels), nrow(iris))
  expect_equal(length(unique(cuda_ml_kclust$labels)), 3L)
  expect_true(all(is.finite(cuda_ml_kclust$centroids)))
  expect_true(is.finite(cuda_ml_kclust$inertia))
  expect_lte(cuda_ml_kclust$n_iter, 100L)
})

test_that("cuda_ml_kmeans() works as expected with user-specified initial cluster centers", {
  cuda_ml_kclust <- cuda_ml_kmeans(
    iris[, which(names(iris) != "Species")],
    k = 3,
    max_iters = 100,
    init_method = sklearn_kclust$cluster_centers_
  )

  verify_cluster_centers(cuda_ml_kclust$centroids)
  expect_equal(cuda_ml_kclust$n_iter, 1)
})
