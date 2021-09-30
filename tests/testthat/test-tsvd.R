context("Truncated SVD")

tsvd_model <- sklearn$decomposition$TruncatedSVD(
  n_components = 2L, algorithm = "arpack"
)
sklearn_tsvd_model <- tsvd_model$fit(sklearn_iris_dataset$data)

cuda_ml_tsvd_model <- cuda_ml_tsvd(iris[1:4], n_components = 2)

test_that("cuda_ml_tsvd() works as expected", {
  expect_equal(
    cuda_ml_tsvd_model$components, sklearn_tsvd_model$components_,
    tolerance = 1e-8, scale = 1
  )
  expect_equal(
    cuda_ml_tsvd_model$explained_variance,
    as.numeric(sklearn_tsvd_model$explained_variance_),
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
  expect_equal(
    cuda_ml_tsvd_model$transformed_data,
    sklearn_tsvd_model$transform(sklearn_iris_dataset$data),
    tolerance = 1e-8, scale = 1
  )
})

test_that("cuda_ml_inverse_transform() works as expected for TSVD models", {
  expect_equal(
    cuda_ml_inverse_transform(
      cuda_ml_tsvd_model, cuda_ml_tsvd_model$transformed_data
    ),
    sklearn_tsvd_model$inverse_transform(cuda_ml_tsvd_model$transformed_data)
  )
})
