skip_if_not(run_gpu_tests, "requires the GPU test environment")

context("K-Nearest Neighbors")

set.seed(0)
blob_sz <- 1000
centers <- list(c(3, 3), c(-3, -3), c(-3, 3))
blobs <- gen_blobs(blob_sz, centers)
blobs_df <- blobs |>
  as.data.frame() |>
  cbind(
    label = seq_along(centers) |>
      sapply(\(x) rep(x, blob_sz)) |>
      factor()
  )

test_blob_sz <- 10

test_that("KNN defaults fit and predict", {
  model <- cuda_ml_knn(mpg ~ ., mtcars)
  new_data <- mtcars[1:2, names(mtcars) != "mpg", drop = FALSE]
  predictions <- predict(model, new_data)

  expect_s3_class(model, "cuda_ml_knn")
  expect_named(predictions, ".pred")
  expect_equal(nrow(predictions), nrow(new_data))
})

test_that("KNN classifier works as expected", {
  test_blobs_df <- gen_blobs(test_blob_sz, centers) |>
    as.data.frame()
  algos <- c("brute", "ivfflat", "ivfpq")

  for (algo in algos) {
    model <- cuda_ml_knn(
      label ~ .,
      blobs_df,
      algo = algo,
      metric = "euclidean"
    )
    preds <- predict(model, test_blobs_df, type = "class")

    expect_equal(
      as.integer(preds$.pred_class),
      seq(3) |>
        purrr::map(\(x) rep(x, test_blob_sz)) |>
        unlist(use.names = FALSE),
      label = algo
    )
  }
})

test_that("KNN classifier returns per-class probabilities", {
  test_data <- as.data.frame(gen_blobs(test_blob_sz, centers))
  model <- cuda_ml_knn(label ~ ., blobs_df, algo = "brute", neighbors = 5L)

  probabilities <- predict(model, test_data, type = "prob")

  expect_named(probabilities, paste0(".pred_", levels(blobs_df$label)))
  expect_equal(rowSums(probabilities), rep(1, nrow(test_data)))
})

test_that("KNN forwards explicit IVFPQ bit width", {
  test_data <- as.data.frame(gen_blobs(test_blob_sz, centers))
  specification <- cuda_ml_knn_algo_ivfpq(
    nlist = 8L,
    nprobe = 3L,
    m = 2L,
    n_bits = 4L
  )
  model <- cuda_ml_knn(
    label ~ .,
    blobs_df,
    algo = specification,
    metric = "euclidean"
  )

  predictions <- predict(model, test_data, type = "class")

  expect_named(predictions, ".pred_class")
  expect_equal(nrow(predictions), nrow(test_data))
})

test_that("automated IVFPQ selects an aligned bit width", {
  set.seed(1L)
  data <- data.frame(
    x1 = rnorm(1000L),
    x2 = rnorm(1000L),
    x3 = rnorm(1000L),
    label = factor(rep(1:2, each = 500L))
  )

  model <- cuda_ml_knn(
    label ~ .,
    data,
    algo = "ivfpq",
    metric = "euclidean"
  )

  expect_s3_class(model, "cuda_ml_knn")
})

test_that("KNN regressor works as expected", {
  resps <- seq_along(centers) |>
    sapply(\(x) rep(exp(-x), blob_sz)) |>
    c()
  train_df <- blobs |>
    as.data.frame() |>
    cbind(y = resps)
  test_blobs <- gen_blobs(test_blob_sz, centers)

  cuda_ml_knn_regressor <- cuda_ml_knn(
    y ~ .,
    data = train_df,
    algo = "brute",
    metric = "euclidean",
    neighbors = 5L
  )
  cuda_ml_knn_regressor_preds <- predict(
    cuda_ml_knn_regressor,
    as.data.frame(test_blobs)
  )

  sklearn_knn_regressor <- sklearn$neighbors$KNeighborsRegressor(
    n_neighbors = 5L,
    algorithm = "brute",
    metric = "euclidean"
  )
  sklearn_knn_regressor$fit(X = blobs, y = resps)
  sklearn_knn_regressor_preds <- sklearn_knn_regressor$predict(
    as.matrix(test_blobs)
  )

  expect_equal(
    cuda_ml_knn_regressor_preds$.pred,
    as.numeric(sklearn_knn_regressor_preds)
  )
})

test_that("KNN classifier works as expected through parsnip", {
  skip_if_not_installed("parsnip")
  library(parsnip)

  test_blobs_df <- gen_blobs(test_blob_sz, centers) |>
    as.data.frame()
  model <- nearest_neighbor(
    mode = "classification",
    neighbors = 10,
    dist_power = 2
  ) |>
    set_engine("cuda.ml") |>
    fit(label ~ ., blobs_df)
  preds <- predict(model, test_blobs_df)

  expect_equal(
    as.integer(preds$.pred_class),
    seq(3) |>
      purrr::map(\(x) rep(x, test_blob_sz)) |>
      unlist(use.names = FALSE)
  )
})

test_that("KNN regressor works as expected through parsnip", {
  skip_if_not_installed("parsnip")
  library(parsnip)

  resps <- seq_along(centers) |>
    sapply(\(x) rep(exp(-x), blob_sz)) |>
    c()
  train_df <- blobs |>
    as.data.frame() |>
    cbind(y = resps)
  test_blobs <- gen_blobs(test_blob_sz, centers)

  cuda_ml_knn_regressor <- nearest_neighbor(
    mode = "regression",
    neighbors = 5,
    dist_power = 2
  ) |>
    set_engine("cuda.ml") |>
    fit(y ~ ., data = train_df)
  cuda_ml_knn_regressor_preds <- predict(
    cuda_ml_knn_regressor,
    as.data.frame(test_blobs)
  )

  sklearn_knn_regressor <- sklearn$neighbors$KNeighborsRegressor(
    n_neighbors = 5L,
    algorithm = "brute",
    metric = "euclidean"
  )
  sklearn_knn_regressor$fit(X = blobs, y = resps)
  sklearn_knn_regressor_preds <- sklearn_knn_regressor$predict(
    as.matrix(test_blobs)
  )

  expect_equal(
    cuda_ml_knn_regressor_preds$.pred,
    as.numeric(sklearn_knn_regressor_preds)
  )
})
