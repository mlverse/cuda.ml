context("(de)serialization of Random Projection models")

test_that("random projection model can be serialized and unserialized correctly", {
  has_mlbench <- require("mlbench")
  stopifnot(has_mlbench)

  data(Vehicle)
  data <- Vehicle[, which(names(Vehicle) != "Class")]

  model <- cuda_ml_rand_proj(data, n_components = 4)
  model_state <- cuda_ml_serialize(model)

  actual_transformed_data <- callr::r(
    function(model_state, data) {
      library(cuda.ml)

      model <- cuda_ml_unserialize(model_state)

      cuda_ml_transform(model, data)
    },
    args = list(
      model_state = model_state,
      data = data
    )
  )

  expect_equal(actual_transformed_data, model$transformed_data)
})
