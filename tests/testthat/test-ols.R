context("Ordinary Least Square")

test_that("OLS regressor works as expected", {
  for (normalize_input in c(FALSE, TRUE)) {
    for (fit_intercept in c(FALSE, TRUE)) {
      if (!fit_intercept) {
        input <- scale(mtcars, scale = FALSE)
      } else {
        input <- as.matrix(mtcars)
      }

      if (normalize_input) {
        sklearn_scaler <- sklearn$preprocessing$StandardScaler(
          copy = TRUE, with_mean = TRUE, with_std = TRUE
        )
        sklearn_scaler$fit(as.matrix(input))
        sklearn_input <- sklearn_scaler$transform(as.matrix(input))
      } else {
        sklearn_input <- as.matrix(input)
      }

      sklearn_ols_regressor <- sklearn$linear_model$LinearRegression(
        fit_intercept = fit_intercept
      )
      sklearn_predictors <- sklearn_input[, which(names(mtcars) != "mpg")]
      sklearn_ols_regressor$fit(
        X = sklearn_predictors,
        y = mtcars$mpg
      )
      sklearn_ols_regressor_preds <- sklearn_ols_regressor$predict(
        sklearn_input[, which(names(mtcars) != "mpg")]
      )

      for (algo in c("svd", "eig", "qr")) {
        cuda_ml_ols_regressor <- cuda_ml_ols(
          mpg ~ ., input,
          algorithm = algo,
          fit_intercept = fit_intercept,
          normalize_input = normalize_input
        )
        cuda_ml_ols_regressor_preds <- predict(
          cuda_ml_ols_regressor, input[, which(names(mtcars) != "mpg")]
        )

        expect_equal(
          cuda_ml_ols_regressor_preds$.pred,
          as.numeric(sklearn_ols_regressor_preds)
        )
      }
    }
  }
})
