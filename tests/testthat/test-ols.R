skip_if_not(run_gpu_tests, "requires the GPU test environment")

test_that("OLS regressor works as expected", {
  for (fit_intercept in c(FALSE, TRUE)) {
    input <- as.matrix(mtcars)
    if (!fit_intercept) {
      input <- scale(input, scale = FALSE)
    }

    predictors <- input[, names(mtcars) != "mpg", drop = FALSE]
    reference <- sklearn$linear_model$LinearRegression(
      fit_intercept = fit_intercept
    )
    reference$fit(X = predictors, y = mtcars$mpg)
    expected <- reference$predict(predictors)

    for (method in c("svd", "eig", "qr")) {
      model <- cuda_ml_ols(
        mpg ~ .,
        input,
        method = method,
        fit_intercept = fit_intercept
      )

      expect_equal(
        predict(model, predictors)$.pred,
        as.numeric(expected)
      )
    }
  }
})
