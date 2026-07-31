test_that("only current KNN algorithms are exposed", {
  exports <- getNamespaceExports("cuda.ml")

  expect_true("cuda_ml_knn_algo_ivfflat" %in% exports)
  expect_true("cuda_ml_knn_algo_ivfpq" %in% exports)
  expect_false("cuda_ml_knn_algo_ivfsq" %in% exports)
})

test_that("obsolete random projection and FIL interfaces are absent", {
  exports <- getNamespaceExports("cuda.ml")

  expect_false("cuda_ml_rand_proj" %in% exports)
  expect_false("cuda_ml_rand_proj_enabled" %in% exports)
  expect_false("cuda_ml_fil_load_model" %in% exports)
  expect_false("cuda_ml_fil_enabled" %in% exports)
})

test_that("prediction methods use the R modeling new_data convention", {
  methods <- c(
    "predict.cuda_ml_knn",
    "predict.cuda_ml_linear_model",
    "predict.cuda_ml_logistic_reg",
    "predict.cuda_ml_nvforest",
    "predict.cuda_ml_svm"
  )

  for (method in methods) {
    expect_true("new_data" %in% names(formals(get(method))))
  }
})

test_that("classification rejects unobserved outcome levels", {
  data <- iris[iris$Species != "virginica", ]

  expect_error(
    cuda_ml_rand_forest(Species ~ ., data, trees = 10L),
    "Every outcome factor level must be represented"
  )
  expect_error(
    cuda_ml_logistic_reg(Species ~ ., data),
    "Every outcome factor level must be represented"
  )
  expect_error(
    cuda_ml_knn(Species ~ ., data, algo = "brute", metric = "euclidean"),
    "Every outcome factor level must be represented"
  )
  expect_error(
    cuda_ml_svm(Species ~ ., data, cost = 1),
    "Every outcome factor level must be represented"
  )
})

test_that("KNN defaults select current cuML choices", {
  skip_if(cuda_ml_backend_info()$backend == "full")

  expect_error(
    cuda_ml_knn(mpg ~ ., mtcars),
    "CRAN-compatible stub"
  )
})

test_that("KNN parameters fail before backend execution", {
  expect_error(cuda_ml_knn_algo_ivfflat(0, 1), "nlist")
  expect_error(cuda_ml_knn_algo_ivfflat(4, 5), "nprobe")
  expect_error(
    cuda_ml_knn_algo_ivfpq(4, 2, 2, 0),
    "n_bits"
  )
  expect_error(
    cuda_ml_knn_algo_ivfpq(4, 2, 2, 4, NA),
    "use_precomputed_tables"
  )

  specification <- cuda_ml_knn_algo_ivfpq(4, 2, 3, 4)
  expect_s3_class(specification, "cuda_ml_knn_algo")
  expect_error(
    cuda_ml_knn(mpg ~ ., mtcars, algo = specification),
    "m.*divide"
  )
  expect_error(cuda_ml_knn(mpg ~ ., mtcars, p = 0), "p")
  expect_error(
    cuda_ml_knn(mpg ~ ., mtcars, neighbors = nrow(mtcars) + 1L),
    "neighbors"
  )
  expect_error(
    cuda_ml_knn(mpg ~ ., mtcars, algo = list(type = 1L, params = list())),
    "algorithm specification"
  )
})

test_that("random forest arguments fail before native execution", {
  expect_error(
    cuda_ml_rand_forest(Species ~ ., iris, split_criterion = "mse"),
    "one of"
  )
  expect_error(
    cuda_ml_rand_forest(mpg ~ ., mtcars, split_criterion = "gini"),
    "one of"
  )
  expect_error(
    cuda_ml_rand_forest(mpg ~ ., mtcars, split_criterion = "mae"),
    "one of"
  )
  expect_error(
    cuda_ml_rand_forest(mpg ~ ., mtcars, n_streams = 0L),
    "n_streams"
  )
  expect_error(
    cuda_ml_rand_forest(mpg ~ ., mtcars, seed = NA_integer_),
    "seed"
  )
})

test_that("logistic regression arguments fail before native execution", {
  data <- iris[iris$Species != "virginica", ]
  data$Species <- droplevels(data$Species)

  expect_error(
    cuda_ml_logistic_reg(Species ~ ., data, fit_intercept = NA),
    "fit_intercept"
  )
  expect_error(
    cuda_ml_logistic_reg(Species ~ ., data, tol = 0),
    "tol"
  )
  expect_error(
    cuda_ml_logistic_reg(Species ~ ., data, max_iter = 1.5),
    "max_iter"
  )
})
