context("(de)serialization of Random Forest models")

test_that("random forest classifier can be serialized and unserialized correctly", {
  model <- cuda_ml_rand_forest(formula = Species ~ ., data = iris, trees = 200)
  model_state <- cuda_ml_serialize(model)

  data <- iris[-which(names(iris) == "Species")]

  expected_preds <- predict(model, data)
  actual_preds <- predict_in_sub_proc(
    model_state,
    data = data,
    expected_mode = "classification",
    expected_model_cls = "cuda_ml_rand_forest"
  )

  expect_equal(expected_preds, actual_preds)

  if (as.integer(cuML_minor_version()) >= 8) {
    # class probabilities output was not supported in earlier versions of RAPIDS
    # cuML
    expected_cls_probs <- predict(
      model, data,
      output_class_probabilities = TRUE
    )
    actual_cls_probs <- predict_in_sub_proc(
      model_state,
      data = data,
      expected_mode = "classification",
      expected_model_cls = "cuda_ml_rand_forest",
      additional_predict_args = list(output_class_probabilities = TRUE)
    )
    expect_equal(expected_cls_probs, actual_cls_probs, tolerance = 1e-3, scale = 1)
  }
})

test_that("random forest regressor can be serialized and unserialized correctly", {
  model <- cuda_ml_rand_forest(formula = mpg ~ ., data = mtcars, trees = 200)
  model_state <- cuda_ml_serialize(model)

  data <- mtcars[-which(names(mtcars) == "mpg")]

  expected_preds <- predict(model, data)
  actual_preds <- predict_in_sub_proc(
    model_state,
    data = data,
    expected_mode = "regression",
    expected_model_cls = "cuda_ml_rand_forest"
  )

  expect_equal(expected_preds, actual_preds, tolerance = 1e-4, scale = 1)
})
