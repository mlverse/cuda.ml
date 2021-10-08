context("(de)serialization of TSNE models")

test_that("TSNE models can be serialized and unserialized correctly", {
  embedding <- cuda_ml_tsne(iris[1:4], method = "exact")
  model_state <- cuda_ml_serialize(embedding)

  expect_equal(
    embedding,
    callr::r(
      function(model_state) {
        library(cuda.ml)

        stopifnot(has_cuML())
        cuda_ml_unserialize(model_state)
      },
      args = list(model_state = model_state)
    )
  )
})
