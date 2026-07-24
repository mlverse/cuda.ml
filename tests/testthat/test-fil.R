test_that("FIL rejects unsupported loading controls", {
  unsupported_controls <- list(
    algo = "naive",
    storage_type = "dense",
    threads_per_tree = 2L,
    n_items = 1L,
    blocks_per_sm = 1L
  )

  for (control in names(unsupported_controls)) {
    args <- list(
      filename = "unused.json",
      mode = "classification"
    )
    args[[control]] <- unsupported_controls[[control]]

    expect_error(
      do.call(cuda_ml_fil_load_model, args),
      "Only the default FIL loading controls are supported"
    )
  }
})

test_that("FIL returns probabilities for max-index multiclass models", {
  skip_if_not(cuda_ml_fil_enabled())
  skip_if_not_installed("xgboost")

  x <- unname(as.matrix(iris[, names(iris) != "Species"]))
  y <- as.integer(iris$Species) - 1L
  dtrain <- xgboost::xgb.DMatrix(x, label = y)
  xgb_model <- xgboost::xgb.train(
    params = list(
      objective = "multi:softmax",
      num_class = 3L,
      max_depth = 3L,
      eta = 0.3
    ),
    data = dtrain,
    nrounds = 10L,
    verbose = 0L
  )

  model_path <- tempfile(fileext = ".json")
  on.exit(unlink(model_path))
  xgboost::xgb.save(xgb_model, model_path)

  margins <- predict(xgb_model, dtrain, outputmargin = TRUE)
  if (is.null(dim(margins))) {
    margins <- matrix(margins, ncol = 3L, byrow = TRUE)
  } else {
    margins <- as.matrix(margins)
  }
  expected <- exp(margins - apply(margins, 1L, max))
  expected <- expected / rowSums(expected)

  fil_model <- cuda_ml_fil_load_model(
    model_path,
    mode = "classification"
  )
  actual <- predict(
    fil_model,
    x,
    output_class_probabilities = TRUE
  )
  actual <- unname(as.matrix(actual))

  expect_equal(actual, expected, tolerance = 1e-5)
  expect_equal(rowSums(actual), rep(1, nrow(x)), tolerance = 1e-6)
})
