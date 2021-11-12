context("Mini-batch Stochastic Gradient Descent")

set.seed(0L)

gen_blobs_df <- function(n_samples) {
  centers <- list(c(0, 2, 2), c(2, -2, 0))
  blob_sz <- n_samples / 2
  X <- gen_blobs(blob_sz = blob_sz, centers = centers)
  y <- c(-1, 1) %>%
    sapply(function(x) rep(x, blob_sz)) %>%
    c()
  df <- cbind(X, y) %>% as.data.frame()

  df
}

test_that("MBSGD works as expected for training linear regressors", {
  train_df <- gen_blobs_df(n_samples = 1000)
  test_df <- gen_blobs_df(n_samples = 100)

  for (learning_rate in c("constant", "adaptive", "invscaling")) {
    for (penalty in c("l1", "l2", "elasticnet")) {
      for (loss in c("squared_loss", "log", "hinge")) {
        model <- cuda_ml_sgd(
          y ~ ., train_df,
          fit_intercept = TRUE,
          loss = loss, penalty = penalty,
          batch_size = 16,
          learning_rate = learning_rate, eta0 = 1e-5,
          tol = 1e-5
        )
        preds <- predict(model, test_df[names(test_df) != "y"])

        expect_lte(sum(sign(preds$.pred) != sign(test_df$y)) / nrow(test_df), 0.01)
      }
    }
  }
})
