skip_if_not(run_gpu_tests, "requires the GPU test environment")

test_that("random forest classifier state preserves classes and probabilities", {
  model <- cuda_ml_rand_forest(Species ~ ., iris, trees = 200L)
  state <- cuda_ml_serialize(model)
  data <- iris[names(iris) != "Species"]

  expect_identical(
    unserialize(state)$model_abi,
    "cuda_ml_rand_forest_model_state"
  )

  expected_class <- predict(model, data, type = "class")
  expected_prob <- predict(model, data, type = "prob")
  restored_class <- predict_in_sub_proc(
    state,
    data,
    expected_mode = "classification",
    expected_model_cls = "cuda_ml_rand_forest",
    additional_predict_args = list(type = "class")
  )
  restored_prob <- predict_in_sub_proc(
    state,
    data,
    expected_mode = "classification",
    expected_model_cls = "cuda_ml_rand_forest",
    additional_predict_args = list(type = "prob")
  )

  expect_equal(restored_class, expected_class)
  expect_equal(restored_prob, expected_prob, tolerance = 1e-3, scale = 1)
})

test_that("random forest regressor state preserves predictions", {
  model <- cuda_ml_rand_forest(mpg ~ ., mtcars, trees = 200L)
  state <- cuda_ml_serialize(model)
  data <- mtcars[names(mtcars) != "mpg"]

  expect_identical(
    unserialize(state)$model_abi,
    "cuda_ml_rand_forest_model_state"
  )

  expected <- predict(model, data)
  restored <- predict_in_sub_proc(
    state,
    data,
    expected_mode = "regression",
    expected_model_cls = "cuda_ml_rand_forest"
  )

  expect_equal(restored, expected, tolerance = 1e-4, scale = 1)
})
