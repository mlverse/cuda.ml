test_that("SVM regressor can be serialized and unserialized correctly", {
  model <- cuml_svm(formula = mpg ~ ., data = mtcars, kernel = "rbf")
  model_state <- cuml_serialize(model)

  expected_preds <- predict(model, mtcars)
  actual_preds <- predict_in_sub_proc(
    model_state,
    data = mtcars,
    expected_mode = "regression",
    expected_model_cls = "cuml_svr"
  )

  expect_equal(expected_preds, actual_preds)
})
