context("k-means")

sklearn_kmeans_model <- sklearn$cluster$KMeans(
  n_clusters = 3L, max_iter = 100L
)
sklearn_kclust <- sklearn_kmeans_model$fit(sklearn_iris_dataset$data)

verify_cluster_centers <- function(centers) {
  expect_equal(
    sort_mat(centers),
    sort_mat(sklearn_kclust$cluster_centers_),
    tol = 0.01,
    scale = 1
  )
}

test_that("cuml_kmeans() works as expected with 'kmeans++' initialization method", {
  cuml_kclust <- cuml_kmeans(
    iris[,which(names(iris) != "Species")],
    k = 3,
    max_iters = 100,
    init_method = "kmeans++"
  )

  verify_cluster_centers(cuml_kclust$centroids)
})

test_that("cuml_kmeans() works as expected with 'random' initialization method", {
  cuml_kclust <- cuml_kmeans(
    iris[,which(names(iris) != "Species")],
    k = 3,
    max_iters = 100,
    init_method = "random"
  )

  verify_cluster_centers(cuml_kclust$centroids)
})

test_that("cuml_kmeans() works as expected with user-specified initial cluster centers", {
  cuml_kclust <- cuml_kmeans(
    iris[,which(names(iris) != "Species")],
    k = 3,
    max_iters = 100,
    init_method = sklearn_kclust$cluster_centers_
  )

  verify_cluster_centers(cuml_kclust$centroids)
  expect_equal(cuml_kclust$n_iter, 1)
})
