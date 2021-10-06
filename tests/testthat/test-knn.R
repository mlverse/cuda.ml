context("K-Nearest Neighbors")

set.seed(0)
blob_sz <- 1000
centers <- list(c(3, 3), c(-3, -3), c(-3, 3))
blobs <- gen_blobs(blob_sz, centers)
blobs_df <- blobs %>%
  as.data.frame() %>%
  cbind(
    label = seq_along(centers) %>%
      sapply(function(x) rep(x, blob_sz)) %>%
      factor()
  )

test_blob_sz <- 10
test_blobs_df <- gen_blobs(test_blob_sz, centers) %>%
  as.data.frame()

test_that("KNN classifier works as expected", {
  for (algo in c("brute", "ivfflat", "ivfpq", "ivfsq")) {
    model <- cuda_ml_knn(
      label ~ ., blobs_df, algo = algo, metric = "euclidean"
    )
    preds <- predict(model, test_blobs_df)

    expect_equal(
      as.integer(preds$.pred_class),
      seq(3) %>%
        purrr::map(~ rep(.x, test_blob_sz)) %>%
        purrr::flatten_int(),
      label = algo
    )
  }
})
