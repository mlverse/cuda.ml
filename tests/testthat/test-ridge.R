context("Ridge Regression")

test_that("Ridge regressor works as expected", {
  for (normalize_input in c(FALSE, TRUE)) {
    for (fit_intercept in c(FALSE, TRUE)) {
      if (!fit_intercept && normalize_input) {
        next
      }

      if (!fit_intercept) {
        input <- mtcars
        input[names(mtcars) != "mpg"] <- scale(
          input[names(mtcars) != "mpg"],
          scale = FALSE
        )
        input <- as.matrix(input)
      } else {
        input <- as.matrix(mtcars)
      }

      if (normalize_input) {
        sklearn_scaler <- sklearn$preprocessing$StandardScaler(
          copy = TRUE, with_mean = TRUE, with_std = TRUE
        )
        sklearn_scaler$fit(as.matrix(mtcars[names(mtcars) != "mpg"]))
        sklearn_predictors <- sklearn_scaler$transform(
          as.matrix(mtcars[names(mtcars) != "mpg"])
        )
      } else {
        sklearn_predictors <- as.matrix(input[, which(names(mtcars) != "mpg")])
      }

      sklearn_ridge_regressor <- sklearn$linear_model$Ridge(
        alpha = 1e-3, fit_intercept = fit_intercept
      )
      sklearn_ridge_regressor$fit(
        X = sklearn_predictors,
        y = mtcars$mpg
      )
      sklearn_ridge_regressor_preds <- sklearn_ridge_regressor$predict(
        sklearn_predictors
      )

      cuda_ml_ridge_regressor <- cuda_ml_ridge(
        mpg ~ ., input,
        alpha = 1e-3,
        fit_intercept = fit_intercept,
        normalize_input = normalize_input
      )
      cuda_ml_ridge_regressor_preds <- predict(
        cuda_ml_ridge_regressor, input
      )

      expect_equal(
        cuda_ml_ridge_regressor_preds$.pred,
        as.numeric(sklearn_ridge_regressor_preds),
        tol = 0.05,
        scale = 1
      )
    }
  }
})
