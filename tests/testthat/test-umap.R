skip_if_not(run_gpu_tests, "requires the GPU test environment")

context("Uniform Manifold Approximation and Projection")

penguins_input <- scaled_penguin_predictors

test_that("cuda_ml_umap() works as expected", {
  umap_output <- cuda_ml_umap(
    x = penguins_input,
    y = penguins$species,
    n_components = 2,
    n_epochs = 500,
    transform_input = TRUE,
    seed = 0L
  )

  verify_penguins_embedding(umap_output$transformed_data)
})
