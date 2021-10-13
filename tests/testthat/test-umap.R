context("Uniform Manifold Approximation and Projection")

iris_input <- iris[, names(iris) != "Species"]

test_that("cuda_ml_umap() works as expected", {
  umap_output <- cuda_ml_umap(
    x = iris_input, y = iris$Species, n_components = 2,
    n_epochs = 500, transform_input = TRUE, seed = 0L
  )

  verify_iris_embedding(umap_output$transformed_data)
})
