context("Least Absolute Shrinkage and Selection Operator")

test_that("LASSO regressor works as expected", {
  for (normalize_input in c(FALSE, TRUE)) {
    for (fit_intercept in c(FALSE, TRUE)) {
      if (!fit_intercept && normalize_input) {
        next
      }

      if (!fit_intercept) {
        input <- mtcars
        input[, which(names(mtcars) != "mpg")] <- scale(
          input[, which(names(mtcars) != "mpg")],
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
        sklearn_scaler$fit(as.matrix(mtcars[, which(names(mtcars) != "mpg")]))
        sklearn_predictors <- sklearn_scaler$transform(
          as.matrix(mtcars[, which(names(mtcars) != "mpg")])
        )
      } else {
        sklearn_predictors <- as.matrix(input[, which(names(mtcars) != "mpg")])
      }

      sklearn_lasso_regressor <- sklearn$linear_model$Lasso(
        alpha = 1e-3,
        max_iter = 10000L,
        tol = 1e-4,
        fit_intercept = fit_intercept
      )
      sklearn_lasso_regressor$fit(
        X = sklearn_predictors,
        y = mtcars$mpg
      )
      sklearn_lasso_regressor_preds <- sklearn_lasso_regressor$predict(
        sklearn_predictors
      )

      cuda_ml_lasso_regressor <- cuda_ml_lasso(
        mpg ~ ., input,
        alpha = 1e-3,
        max_iter = 10000,
        tol = 1e-4,
        fit_intercept = fit_intercept,
        normalize_input = normalize_input
      )
      cuda_ml_lasso_regressor_preds <- predict(
        cuda_ml_lasso_regressor, input
      )

      expect_equal(
        cuda_ml_lasso_regressor_preds$.pred,
        as.numeric(sklearn_lasso_regressor_preds),
        tol = 0.05,
        scale = 1
      )
    }
  }
})

test_that("LASSO normalization is applied before regularization", {
  x <- cbind(
    small = seq(-2, 2, length.out = 20),
    large = rep(c(-1000, 1000), 10)
  )
  y <- 3 * x[, "small"] + 0.002 * x[, "large"] +
    rep(c(-0.1, 0.1), 10)

  x_center <- colMeans(x)
  x_centered <- sweep(x, 2, x_center)
  x_norm <- sqrt(colSums(x_centered^2))
  x_normalized <- sweep(x_centered, 2, x_norm, "/")

  normalized_fit <- cuda_ml_lasso(
    x,
    y,
    alpha = 0.01,
    max_iter = 10000L,
    tol = 1e-8,
    normalize_input = TRUE
  )
  reference_fit <- cuda_ml_lasso(
    x_normalized,
    y,
    alpha = 0.01,
    max_iter = 10000L,
    tol = 1e-8,
    normalize_input = FALSE
  )

  expect_equal(
    predict(normalized_fit, x)$.pred,
    predict(reference_fit, x_normalized)$.pred,
    tolerance = 1e-6,
    scale = 1
  )
})
