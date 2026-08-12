skip_if_not(run_gpu_tests, "requires the GPU test environment")

context("Truncated SVD")

tsvd_model <- sklearn$decomposition$TruncatedSVD(
  n_components = 2L,
  algorithm = "arpack"
)
sklearn_tsvd_model <- tsvd_model$fit(sklearn_penguins_dataset$data)

cuda_ml_tsvd_model <- cuda_ml_tsvd(
  scaled_penguin_predictors,
  n_components = 2
)

# SVD components are only defined up to sign — align signs before comparing.
# For each component row, flip the cuML sign to match sklearn if the first
# non-negligible element disagrees.
align_svd_signs <- function(a, b) {
  for (i in seq_len(nrow(a))) {
    if (sign(a[i, 1]) != sign(b[i, 1])) {
      a[i, ] <- -a[i, ]
    }
  }
  a
}

test_that("cuda_ml_tsvd() works as expected", {
  sklearn_components <- sklearn_tsvd_model$components_
  aligned_components <- align_svd_signs(
    cuda_ml_tsvd_model$components,
    sklearn_components
  )

  expect_equal(
    aligned_components,
    sklearn_components,
    tolerance = 1e-8,
    scale = 1
  )
  expect_equal(
    cuda_ml_tsvd_model$explained_variance,
    as.numeric(sklearn_tsvd_model$explained_variance_),
    tolerance = 1e-8,
    scale = 1
  )
  expect_equal(
    cuda_ml_tsvd_model$explained_variance_ratio,
    as.numeric(sklearn_tsvd_model$explained_variance_ratio_),
    tolerance = 1e-8,
    scale = 1
  )
  expect_equal(
    cuda_ml_tsvd_model$singular_values,
    as.numeric(sklearn_tsvd_model$singular_values_),
    tolerance = 1e-8,
    scale = 1
  )

  # Transformed data columns also have sign ambiguity matching the components
  sklearn_transformed <- sklearn_tsvd_model$transform(
    sklearn_penguins_dataset$data
  )
  cuda_transformed <- cuda_ml_tsvd_model$transformed_data
  for (j in seq_len(ncol(cuda_transformed))) {
    if (sign(cuda_transformed[1, j]) != sign(sklearn_transformed[1, j])) {
      cuda_transformed[, j] <- -cuda_transformed[, j]
    }
  }
  expect_equal(
    cuda_transformed,
    sklearn_transformed,
    tolerance = 1e-8,
    scale = 1
  )
})

test_that("cuda_ml_inverse_transform() works as expected for TSVD models", {
  # inverse_transform recovers the original data regardless of sign convention
  cuda_ml_reconstructed <- cuda_ml_inverse_transform(
    cuda_ml_tsvd_model,
    cuda_ml_tsvd_model$transformed_data
  )
  sklearn_reconstructed <- sklearn_tsvd_model$inverse_transform(
    sklearn_tsvd_model$transform(sklearn_penguins_dataset$data)
  )
  expect_equal(
    cuda_ml_reconstructed,
    sklearn_reconstructed,
    tolerance = 1e-2,
    scale = 1
  )
})

test_that("TSVD transformations use the current batch size", {
  batch_sizes <- c(3L, nrow(sklearn_penguins_dataset$data) + 7L)

  for (batch_size in batch_sizes) {
    rows <- rep(
      seq_len(nrow(sklearn_penguins_dataset$data)),
      length.out = batch_size
    )
    new_data <- sklearn_penguins_dataset$data[rows, , drop = FALSE]
    expected_transformed <- new_data %*% t(cuda_ml_tsvd_model$components)

    transformed <- cuda_ml_transform(cuda_ml_tsvd_model, new_data)

    expect_identical(dim(transformed), c(batch_size, 2L))
    expect_equal(
      transformed,
      expected_transformed,
      tolerance = 1e-8,
      scale = 1
    )

    expected_reconstructed <-
      expected_transformed %*% cuda_ml_tsvd_model$components
    reconstructed <- cuda_ml_inverse_transform(
      cuda_ml_tsvd_model,
      expected_transformed
    )

    expect_identical(dim(reconstructed), c(batch_size, 4L))
    expect_equal(
      reconstructed,
      expected_reconstructed,
      tolerance = 1e-8,
      scale = 1
    )
  }
})

test_that("TSVD transformations reject incompatible input widths", {
  expect_error(
    cuda_ml_transform(
      cuda_ml_tsvd_model,
      cbind(sklearn_penguins_dataset$data, extra = 0)
    ),
    "same number of columns"
  )
  expect_error(
    cuda_ml_inverse_transform(
      cuda_ml_tsvd_model,
      cbind(cuda_ml_tsvd_model$transformed_data, extra = 0)
    ),
    "one column per fitted TSVD component"
  )
})
