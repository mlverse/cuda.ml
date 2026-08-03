skip_if_not(run_gpu_tests, "requires the GPU test environment")

test_that("ridge regressor works as expected", {
  for (fit_intercept in c(FALSE, TRUE)) {
    input <- mtcars
    if (!fit_intercept) {
      input[names(input) != "mpg"] <- scale(
        input[names(input) != "mpg"],
        scale = FALSE
      )
    }
    predictors <- as.matrix(input[names(input) != "mpg"])

    reference <- sklearn$linear_model$Ridge(
      alpha = 1e-3,
      fit_intercept = fit_intercept
    )
    reference$fit(X = predictors, y = mtcars$mpg)

    model <- cuda_ml_ridge(
      mpg ~ .,
      input,
      alpha = 1e-3,
      fit_intercept = fit_intercept
    )

    expect_equal(
      predict(model, input)$.pred,
      as.numeric(reference$predict(predictors)),
      tolerance = 0.05,
      scale = 1
    )
  }
})
