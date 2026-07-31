skip_if_not(run_gpu_tests, "requires the GPU test environment")

test_that("elastic net regressor works as expected", {
  for (fit_intercept in c(FALSE, TRUE)) {
    for (l1_ratio in c(0.4, 0.5, 0.6)) {
      input <- mtcars
      if (!fit_intercept) {
        input[names(input) != "mpg"] <- scale(
          input[names(input) != "mpg"],
          scale = FALSE
        )
      }
      predictors <- as.matrix(input[names(input) != "mpg"])

      reference <- sklearn$linear_model$ElasticNet(
        alpha = 1e-3,
        max_iter = 10000L,
        tol = 1e-4,
        fit_intercept = fit_intercept,
        l1_ratio = l1_ratio
      )
      reference$fit(X = predictors, y = mtcars$mpg)

      model <- cuda_ml_elastic_net(
        mpg ~ .,
        input,
        alpha = 1e-3,
        l1_ratio = l1_ratio,
        max_iter = 10000L,
        tol = 1e-4,
        fit_intercept = fit_intercept
      )

      expect_equal(
        predict(model, input)$.pred,
        as.numeric(reference$predict(predictors)),
        tolerance = 0.3,
        scale = 1
      )
    }
  }
})
