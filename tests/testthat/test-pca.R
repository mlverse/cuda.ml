skip_if_not(run_gpu_tests, "requires the GPU test environment")

context("Principal Component Analysis")

cuda_ml_pca_model <- cuda_ml_pca(
  iris[, which(names(iris) != "Species")],
  n_components = 3
)

sklearn_pca_model <- sklearn$decomposition$PCA(n_components = 3L, copy = TRUE)
sklearn_pca_model$fit(sklearn_iris_dataset$data)

test_that("cuda_ml_pca() works as expected", {
  expect_equal(
    cuda_ml_pca_model$components,
    as.matrix(sklearn_pca_model$components_),
    tolerance = 1e-8,
    scale = 1
  )
  expect_equal(
    cuda_ml_pca_model$explained_variance,
    as.numeric(sklearn_pca_model$explained_variance_),
    tolerance = 1e-8,
    scale = 1
  )
  expect_equal(
    cuda_ml_pca_model$explained_variance_ratio,
    as.numeric(sklearn_pca_model$explained_variance_ratio_),
    tolerance = 1e-8,
    scale = 1
  )
  expect_equal(
    cuda_ml_pca_model$singular_values,
    as.numeric(sklearn_pca_model$singular_values_),
    tolerance = 1e-8,
    scale = 1
  )
  expect_equal(
    cuda_ml_pca_model$mean,
    as.numeric(sklearn_pca_model$mean_),
    tolerance = 1e-8,
    scale = 1
  )
})

test_that("cuda_ml_inverse_transform() works as expected for PCA models", {
  expect_equal(
    cuda_ml_pca_model$transformed_data %>%
      sklearn_pca_model$inverse_transform() %>%
      as.matrix(),
    cuda_ml_pca_model %>%
      cuda_ml_inverse_transform(cuda_ml_pca_model$transformed_data),
    tolerance = 1e-8,
    scale = 1
  )
})

test_that("PCA inverse transformation uses the current batch size", {
  batch_sizes <- c(3L, nrow(cuda_ml_pca_model$transformed_data) + 7L)

  for (batch_size in batch_sizes) {
    rows <- rep(
      seq_len(nrow(cuda_ml_pca_model$transformed_data)),
      length.out = batch_size
    )
    transformed <- cuda_ml_pca_model$transformed_data[rows, , drop = FALSE]
    expected <- sweep(
      transformed %*% cuda_ml_pca_model$components,
      MARGIN = 2L,
      STATS = cuda_ml_pca_model$mean,
      FUN = "+"
    )

    reconstructed <- cuda_ml_inverse_transform(
      cuda_ml_pca_model,
      transformed
    )

    expect_identical(dim(reconstructed), c(batch_size, 4L))
    expect_equal(reconstructed, expected, tolerance = 1e-8, scale = 1)
  }
})

test_that("PCA inverse transformation rejects incompatible input widths", {
  expect_error(
    cuda_ml_inverse_transform(
      cuda_ml_pca_model,
      cbind(cuda_ml_pca_model$transformed_data, extra = 0)
    ),
    "one column per fitted PCA component"
  )
})
