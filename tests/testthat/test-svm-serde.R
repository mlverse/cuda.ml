test_that("SVM regressor can be serialized and unserialized correctly", {
  model <- cuml_svm(formula = mpg ~ ., data = mtcars, kernel = "rbf")
  model_state <- cuml_serialize(model)

  expected_preds <- predict(model, mtcars)
  actual_preds <- callr::r(
    function(model_state) {
      library(cuml)

      model <- cuml_unserialize(model_state)
      stopifnot("cuml_svr" %in% class(model))
      stopifnot(identical(model$mode, "regression"))

      predict(model, mtcars)
    },
    args = list(model_state = model_state),
    stdout = "", stderr = ""
  )

  expect_equal(expected_preds, actual_preds)
})
