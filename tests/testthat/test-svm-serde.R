skip_if_not(run_gpu_tests, "requires the GPU test environment")

context("(de)serialization of SVM models")

test_that("SVM regressor can be serialized and unserialized correctly", {
  model <- cuda_ml_svm(formula = mpg ~ ., data = mtcars, kernel = "rbf")
  model_state <- cuda_ml_serialize(model)

  expected_preds <- predict(model, mtcars)
  actual_preds <- predict_saved_models_in_sub_proc(model, mtcars)

  compatible <- unserialize(model_state)
  provenance <- c(
    "cuda_version",
    "nvforest_version",
    "treelite_version",
    "platform"
  )
  compatible$backend[provenance] <- "different provenance"
  compatible_preds <- predict(
    cuda_ml_unserialize(serialize(compatible, NULL)),
    mtcars
  )

  incompatible <- unserialize(model_state)
  incompatible$backend$rapids_version <- "0.0"
  expect_error(
    cuda_ml_unserialize(serialize(incompatible, NULL)),
    "requires backend `rapids_version`.*0.0"
  )

  missing <- unserialize(model_state)
  missing$backend$rapids_version <- NULL
  expect_error(
    cuda_ml_unserialize(serialize(missing, NULL)),
    "missing required backend identity `rapids_version`"
  )

  expect_equal(expected_preds, compatible_preds)
  expect_equal(expected_preds, actual_preds$restored)
  expect_equal(expected_preds, actual_preds$unbundled)
})
