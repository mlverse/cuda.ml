context("(de)serialization of PCA models")

test_that("PCA models can be serialized and unserialized correctly", {
  models <- list()

  for (whiten in c(FALSE, TRUE)) {
    for (transform_input in c(FALSE, TRUE)) {
      model <- cuda_ml_pca(
        iris[1:4],
        n_components = 3, whiten = whiten,
        transform_input = transform_input
      )
      models <- append(models, list(model))
    }
  }

  for (model in models) {
    actual_inv_transform <- callr::r(
      function(model_state, expected_components, expected_expl_var,
               expected_expl_var_ratio, expected_sg_vals, expected_m,
               expected_tf_data, whiten) {
        library(cuda.ml)
        library(testthat)

        stopifnot(has_cuML())

        model <- cuda_ml_unserialize(model_state)

        expect_equal(model$components, expected_components)
        expect_equal(model$explained_variance, expected_expl_var)
        expect_equal(
          model$explained_variance_ratio, expected_expl_var_ratio
        )
        expect_equal(model$singular_values, expected_sg_vals)
        expect_equal(model$mean, expected_m)
        expect_equal(model$transformed_data, expected_tf_data)

        if (!is.null(model$transformed_data) && !whiten) {
          # TODO: look into why this may fail when `whiten` is `TRUE`
          cuda_ml_inverse_transform(model, model$transformed_data)
        } else {
          NULL
        }
      },
      args = list(
        model_state = cuda_ml_serialize(model),
        expected_components = model$components,
        expected_expl_var = model$explained_variance,
        expected_expl_var_ratio = model$explained_variance_ratio,
        expected_sg_vals = model$singular_values,
        expected_m = model$mean,
        expected_tf_data = model$transformed_data,
        whiten = whiten
      )
    )

    if (!is.null(model$transformed_data) && !whiten) {
      # TODO: look into why this may fail when `whiten` is `TRUE`
      expected_inv_transform <- cuda_ml_inverse_transform(
        model, model$transformed_data
      )

      expect_equal(expected_inv_transform, actual_inv_transform)
    } else {
      succeed()
    }
  }
})
