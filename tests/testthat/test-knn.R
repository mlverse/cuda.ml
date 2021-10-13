context("K-Nearest Neighbors")

set.seed(0)
blob_sz <- 1000
centers <- list(c(3, 3), c(-3, -3), c(-3, 3))
blobs <- gen_blobs(blob_sz, centers)
blobs_df <- blobs %>%
  as.data.frame() %>%
  cbind(
    label = seq_along(centers) %>%
      sapply(function(x) rep(x, blob_sz)) %>%
      factor()
  )

test_blob_sz <- 10

test_that("KNN classifier works as expected", {
  test_blobs_df <- gen_blobs(test_blob_sz, centers) %>%
    as.data.frame()
  for (algo in c("brute", "ivfflat", "ivfpq", "ivfsq")) {
    model <- cuda_ml_knn(
      label ~ ., blobs_df,
      algo = algo, metric = "euclidean"
    )
    preds <- predict(model, test_blobs_df)

    expect_equal(
      as.integer(preds$.pred_class),
      seq(3) %>%
        purrr::map(~ rep(.x, test_blob_sz)) %>%
        purrr::flatten_int(),
      label = algo
    )
  }
})

test_that("KNN regressor works as expected", {
  resps <- seq_along(centers) %>%
    sapply(function(x) rep(exp(-x), blob_sz)) %>%
    c()
  train_df <- blobs %>%
    as.data.frame() %>%
    cbind(y = resps)
  test_blobs <- gen_blobs(test_blob_sz, centers)

  cuda_ml_knn_regressor <- cuda_ml_knn(
    y ~ .,
    data = train_df, algo = "brute", metric = "euclidean", neighbors = 5L
  )
  cuda_ml_knn_regressor_preds <- predict(
    cuda_ml_knn_regressor, as.data.frame(test_blobs)
  )

  sklearn_knn_regressor <- sklearn$neighbors$KNeighborsRegressor(
    n_neighbors = 5L, algorithm = "brute", metric = "euclidean"
  )
  sklearn_knn_regressor$fit(X = blobs, y = resps)
  sklearn_knn_regressor_preds <- sklearn_knn_regressor$predict(
    as.matrix(test_blobs)
  )

  expect_equal(
    cuda_ml_knn_regressor_preds$.pred, as.numeric(sklearn_knn_regressor_preds)
  )
})

test_that("KNN classifier works as expected through parsnip", {
  require("parsnip")

  test_blobs_df <- gen_blobs(test_blob_sz, centers) %>%
    as.data.frame()
  model <- nearest_neighbor(
    mode = "classification", neighbors = 10, dist_power = 2
  ) %>%
    set_engine("cuda.ml") %>%
    fit(label ~ ., blobs_df)
  preds <- predict(model, test_blobs_df)

  expect_equal(
    as.integer(preds$.pred_class),
    seq(3) %>%
      purrr::map(~ rep(.x, test_blob_sz)) %>%
      purrr::flatten_int()
  )
})

test_that("KNN regressor works as expected through parsnip", {
  require("parsnip")

  resps <- seq_along(centers) %>%
    sapply(function(x) rep(exp(-x), blob_sz)) %>%
    c()
  train_df <- blobs %>%
    as.data.frame() %>%
    cbind(y = resps)
  test_blobs <- gen_blobs(test_blob_sz, centers)

  cuda_ml_knn_regressor <- nearest_neighbor(
    mode = "regression", neighbors = 5, dist_power = 2
  ) %>%
    set_engine("cuda.ml") %>%
    fit(y ~ ., data = train_df)
  cuda_ml_knn_regressor_preds <- predict(
    cuda_ml_knn_regressor, as.data.frame(test_blobs)
  )

  sklearn_knn_regressor <- sklearn$neighbors$KNeighborsRegressor(
    n_neighbors = 5L, algorithm = "brute", metric = "euclidean"
  )
  sklearn_knn_regressor$fit(X = blobs, y = resps)
  sklearn_knn_regressor_preds <- sklearn_knn_regressor$predict(
    as.matrix(test_blobs)
  )

  expect_equal(
    cuda_ml_knn_regressor_preds$.pred, as.numeric(sklearn_knn_regressor_preds)
  )
})
