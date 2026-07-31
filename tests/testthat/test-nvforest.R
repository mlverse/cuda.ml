skip_if_not(run_gpu_tests, "requires the GPU test environment")
skip_if_not_installed("xgboost")

test_that("nvForest loads current XGBoost formats and reports model metadata", {
  x <- unname(as.matrix(iris[names(iris) != "Species"]))
  y <- as.integer(iris$Species) - 1L
  training <- xgboost::xgb.DMatrix(x, label = y)
  xgb_model <- xgboost::xgb.train(
    params = list(
      objective = "multi:softmax",
      num_class = 3L,
      max_depth = 3L,
      eta = 0.3
    ),
    data = training,
    nrounds = 10L,
    verbose = 0L
  )
  path <- tempfile(fileext = ".json")
  on.exit(unlink(path))
  xgboost::xgb.save(xgb_model, path)

  model <- cuda_ml_nvforest_load_model(
    path,
    model_type = "xgboost_json",
    class_levels = levels(iris$Species),
    device = "gpu"
  )
  info <- cuda_ml_nvforest_info(model)
  classes <- predict(model, x, type = "class")
  per_tree <- cuda_ml_nvforest_predict_per_tree(model, x)

  expect_identical(info$task_type, "multiclass_classification")
  expect_identical(info$device, "gpu")
  expect_identical(info$align_bytes, 0L)
  expect_named(classes, ".pred_class")
  if (info$has_vector_leaves) {
    expect_identical(
      dim(per_tree),
      c(nrow(x), info$num_trees, info$num_outputs)
    )
  } else {
    expect_identical(dim(per_tree), c(nrow(x), info$num_trees))
  }
  expect_error(
    predict(model, x, type = "prob"),
    "probability"
  )
})

test_that("nvForest exposes leaf and per-tree predictions", {
  x <- unname(as.matrix(mtcars[names(mtcars) != "mpg"]))
  training <- xgboost::xgb.DMatrix(x, label = mtcars$mpg)
  xgb_model <- xgboost::xgb.train(
    params = list(
      objective = "reg:squarederror",
      max_depth = 3L,
      base_score = 0
    ),
    data = training,
    nrounds = 5L,
    verbose = 0L
  )
  path <- tempfile(fileext = ".ubj")
  on.exit(unlink(path))
  xgboost::xgb.save(xgb_model, path)

  model <- cuda_ml_nvforest_load_model(
    path,
    model_type = "xgboost_ubj",
    device = "gpu"
  )
  predictions <- predict(model, x)
  leaves <- cuda_ml_nvforest_leaf_ids(model, x)
  per_tree <- cuda_ml_nvforest_predict_per_tree(model, x)
  info <- cuda_ml_nvforest_info(model)
  expected <- as.numeric(predict(xgb_model, x))
  expected_leaves <- predict(
    xgb_model,
    x,
    predleaf = TRUE,
    strict_shape = FALSE
  )
  storage.mode(expected_leaves) <- "integer"

  expect_equal(predictions$.pred, expected, tolerance = 1e-6, scale = 1)
  expect_equal(rowSums(per_tree), expected, tolerance = 1e-6, scale = 1)
  expect_identical(leaves, expected_leaves)
  expect_identical(dim(per_tree), c(nrow(x), info$num_trees))
})

test_that("nvForest matches XGBoost binary probabilities and classes", {
  data <- iris[iris$Species != "virginica", ]
  class_levels <- levels(droplevels(data$Species))
  x <- unname(as.matrix(data[names(data) != "Species"]))
  y <- as.integer(data$Species) - 1L
  training <- xgboost::xgb.DMatrix(x, label = y)
  xgb_model <- xgboost::xgb.train(
    params = list(objective = "binary:logistic", max_depth = 3L),
    data = training,
    nrounds = 5L,
    verbose = 0L
  )
  path <- tempfile(fileext = ".ubj")
  on.exit(unlink(path))
  xgboost::xgb.save(xgb_model, path)

  model <- cuda_ml_nvforest_load_model(
    path,
    model_type = "xgboost_ubj",
    class_levels = class_levels,
    device = "gpu"
  )
  expected_probability <- drop(
    predict(xgb_model, x, strict_shape = TRUE)
  )
  expected_class <- factor(
    class_levels[1L + (expected_probability >= 0.5)],
    levels = class_levels
  )
  probabilities <- predict(model, x, type = "prob")
  classes <- predict(model, x, type = "class")

  expect_equal(
    unname(as.matrix(probabilities)),
    cbind(1 - expected_probability, expected_probability),
    tolerance = 1e-6,
    scale = 1
  )
  expect_identical(classes$.pred_class, expected_class)
})

test_that("nvForest matches XGBoost multiclass probabilities and classes", {
  x <- unname(as.matrix(iris[names(iris) != "Species"]))
  y <- as.integer(iris$Species) - 1L
  training <- xgboost::xgb.DMatrix(x, label = y)
  xgb_model <- xgboost::xgb.train(
    params = list(
      objective = "multi:softprob",
      num_class = 3L,
      max_depth = 3L
    ),
    data = training,
    nrounds = 5L,
    verbose = 0L
  )
  path <- tempfile(fileext = ".ubj")
  on.exit(unlink(path))
  xgboost::xgb.save(xgb_model, path)

  model <- cuda_ml_nvforest_load_model(
    path,
    model_type = "xgboost_ubj",
    class_levels = levels(iris$Species),
    device = "gpu"
  )
  expected_probability <- predict(xgb_model, x, strict_shape = TRUE)
  expected_class <- factor(
    levels(iris$Species)[max.col(expected_probability, ties.method = "first")],
    levels = levels(iris$Species)
  )
  probabilities <- predict(model, x, type = "prob")
  classes <- predict(model, x, type = "class")

  expect_equal(
    unname(as.matrix(probabilities)),
    unname(expected_probability),
    tolerance = 1e-6,
    scale = 1
  )
  expect_identical(classes$.pred_class, expected_class)
})

test_that("nvForest rejects raw margins as classes or probabilities", {
  data <- iris[iris$Species != "virginica", ]
  x <- unname(as.matrix(data[names(data) != "Species"]))
  y <- as.integer(data$Species) - 1L
  training <- xgboost::xgb.DMatrix(x, label = y)
  xgb_model <- xgboost::xgb.train(
    params = list(objective = "binary:logitraw", max_depth = 3L),
    data = training,
    nrounds = 5L,
    verbose = 0L
  )
  path <- tempfile(fileext = ".ubj")
  on.exit(unlink(path))
  xgboost::xgb.save(xgb_model, path)

  model <- cuda_ml_nvforest_load_model(
    path,
    model_type = "xgboost_ubj",
    class_levels = levels(droplevels(data$Species)),
    device = "gpu"
  )
  info <- cuda_ml_nvforest_info(model)

  expect_false(info$has_probability_output)
  expect_error(predict(model, x, type = "class"), "identity")
  expect_error(predict(model, x, type = "prob"), "identity")
})

test_that("nvForest treats hinge output as classes, not probabilities", {
  data <- iris[iris$Species != "virginica", ]
  x <- unname(as.matrix(data[names(data) != "Species"]))
  y <- as.integer(data$Species) - 1L
  training <- xgboost::xgb.DMatrix(x, label = y)
  xgb_model <- xgboost::xgb.train(
    params = list(objective = "binary:hinge", max_depth = 3L),
    data = training,
    nrounds = 5L,
    verbose = 0L
  )
  path <- tempfile(fileext = ".ubj")
  on.exit(unlink(path))
  xgboost::xgb.save(xgb_model, path)

  model <- cuda_ml_nvforest_load_model(
    path,
    model_type = "xgboost_ubj",
    class_levels = levels(droplevels(data$Species)),
    device = "gpu"
  )
  expected <- factor(
    levels(droplevels(data$Species))[as.integer(predict(xgb_model, x)) + 1L],
    levels = levels(droplevels(data$Species))
  )

  expect_identical(predict(model, x)$.pred_class, expected)
  expect_error(predict(model, x, type = "prob"), "hinge")
  expect_error(predict(model, x, threshold = 0.25), "threshold")
})

test_that("nvForest inspection functions reject other cuda.ml models", {
  model <- cuda_ml_ols(mpg ~ ., mtcars)
  x <- mtcars[names(mtcars) != "mpg"]

  expect_error(cuda_ml_nvforest_info(model), "nvForest-backed")
  expect_error(cuda_ml_nvforest_leaf_ids(model, x), "nvForest-backed")
  expect_error(
    cuda_ml_nvforest_predict_per_tree(model, x),
    "nvForest-backed"
  )
})
