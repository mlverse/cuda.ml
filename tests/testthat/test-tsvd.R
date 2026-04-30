context("Truncated SVD")

tsvd_model <- sklearn$decomposition$TruncatedSVD(
  n_components = 2L, algorithm = "arpack"
)
sklearn_tsvd_model <- tsvd_model$fit(sklearn_iris_dataset$data)

cuda_ml_tsvd_model <- cuda_ml_tsvd(iris[1:4], n_components = 2)

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
  aligned_components <- align_svd_signs(cuda_ml_tsvd_model$components, sklearn_components)

  expect_equal(
    aligned_components, sklearn_components,
    tolerance = 1e-8, scale = 1
  )
  expect_equal(
    cuda_ml_tsvd_model$explained_variance,
    as.numeric(sklearn_tsvd_model$explained_variance_),
    tolerance = 1e-8, scale = 1
  )
  expect_equal(
    cuda_ml_tsvd_model$explained_variance_ratio,
    as.numeric(sklearn_tsvd_model$explained_variance_ratio_),
    tolerance = 1e-8, scale = 1
  )
  expect_equal(
    cuda_ml_tsvd_model$singular_values,
    as.numeric(sklearn_tsvd_model$singular_values_),
    tolerance = 1e-8, scale = 1
  )

  # Transformed data columns also have sign ambiguity matching the components
  sklearn_transformed <- sklearn_tsvd_model$transform(sklearn_iris_dataset$data)
  cuda_transformed <- cuda_ml_tsvd_model$transformed_data
  for (j in seq_len(ncol(cuda_transformed))) {
    if (sign(cuda_transformed[1, j]) != sign(sklearn_transformed[1, j])) {
      cuda_transformed[, j] <- -cuda_transformed[, j]
    }
  }
  expect_equal(cuda_transformed, sklearn_transformed, tolerance = 1e-8, scale = 1)
})

test_that("cuda_ml_inverse_transform() works as expected for TSVD models", {
  # inverse_transform recovers the original data regardless of sign convention
  cuda_ml_reconstructed <- cuda_ml_inverse_transform(
    cuda_ml_tsvd_model, cuda_ml_tsvd_model$transformed_data
  )
  sklearn_reconstructed <- sklearn_tsvd_model$inverse_transform(
    sklearn_tsvd_model$transform(sklearn_iris_dataset$data)
  )
  expect_equal(cuda_ml_reconstructed, sklearn_reconstructed, tolerance = 1e-2, scale = 1)
})
