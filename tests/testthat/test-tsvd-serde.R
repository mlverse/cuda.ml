test_that("TSVD models can be serialized and unserialized correctly", {
  input_data <- iris[1:4]
  model <- cuda_ml_tsvd(input_data, n_components = 2)
  expected_tf_data <- model$transformed_data
  expected_inv_tf_data <- cuda_ml_inverse_transform(model, expected_tf_data)
  model_state <- cuda_ml_serialize(model)

  summary <- callr::r(
    function(model_state, input_data) {
      library(cuda.ml)

      stopifnot(has_libcuml())

      model <- cuda_ml_unserialize(model_state)
      tf_data <- cuda_ml_transform(model, input_data)

      list(
        cls = class(model),
        components = model$components,
        singular_values = model$singular_values,
        tf_data = tf_data,
        inv_tf_data = cuda_ml_inverse_transform(model, tf_data)
      )
    },
    args = list(
      model_state = model_state,
      input_data = input_data
    )
  )

  expect_equal(summary$cls, class(model))
  expect_equal(summary$components, model$components)
  expect_equal(summary$singular_values, model$singular_values)
  expect_equal(summary$tf_data, expected_tf_data)
  expect_equal(summary$inv_tf_data, expected_inv_tf_data)
})
