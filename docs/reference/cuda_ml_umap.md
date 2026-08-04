# Uniform Manifold Approximation and Projection (UMAP) for dimension reduction.

Run the Uniform Manifold Approximation and Projection (UMAP) algorithm
to find a low dimensional embedding of the input data that approximates
an underlying manifold.

## Usage

``` r
cuda_ml_umap(
  x,
  y = NULL,
  n_components = 2L,
  n_neighbors = 15L,
  n_epochs = 500L,
  learning_rate = 1,
  init = c("spectral", "random"),
  min_dist = 0.1,
  spread = 1,
  set_op_mix_ratio = 1,
  local_connectivity = 1L,
  repulsion_strength = 1,
  negative_sample_rate = 5L,
  transform_queue_size = 4,
  a = NULL,
  b = NULL,
  target_n_neighbors = n_neighbors,
  target_metric = c("categorical", "euclidean"),
  target_weight = 0.5,
  transform_input = TRUE,
  seed = NULL
)
```

## Arguments

- x:

  The input matrix or dataframe. Each data point should be a row and
  should consist of numeric values only.

- y:

  An optional numeric vector of target values for supervised dimension
  reduction. Default: NULL.

- n_components:

  The dimension of the space to embed into. Default: 2.

- n_neighbors:

  The size of local neighborhood (in terms of number of neighboring
  sample points) used for manifold approximation. Default: 15.

- n_epochs:

  The number of training epochs to be used in optimizing the low
  dimensional embedding. Default: 500.

- learning_rate:

  The initial learning rate for the embedding optimization. Default:
  1.0.

- init:

  Initialization mode of the low dimensional embedding. Must be one of
  {"spectral", "random"}. Default: "spectral".

- min_dist:

  The effective minimum distance between embedded points. Default: 0.1.

- spread:

  The effective scale of embedded points. In combination with `min_dist`
  this determines how clustered/clumped the embedded points are.
  Default: 1.0.

- set_op_mix_ratio:

  Interpolate between (fuzzy) union and intersection as the set
  operation used to combine local fuzzy simplicial sets to obtain a
  global fuzzy simplicial sets. Both fuzzy set operations use the
  product t-norm. The value of this parameter should be between 0.0 and
  1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a
  pure fuzzy intersection. Default: 1.0.

- local_connectivity:

  The local connectivity required – i.e. the number of nearest neighbors
  that should be assumed to be connected at a local level. Default: 1.

- repulsion_strength:

  Weighting applied to negative samples in low dimensional embedding
  optimization. Values higher than one will result in greater weight
  being given to negative samples. Default: 1.0.

- negative_sample_rate:

  The number of negative samples to select per positive sample in the
  optimization process. Default: 5.

- transform_queue_size:

  For transform operations (embedding new points using a trained model),
  this controls how aggressively to search for nearest neighbors.
  Default: 4.0.

- a, b:

  More specific parameters controlling the embedding. If not set, then
  these values are set automatically as determined by `min_dist` and
  `spread`. Default: NULL.

- target_n_neighbors:

  The number of nearest neighbors to use to construct the target
  simplicial set. Default: n_neighbors.

- target_metric:

  The metric for measuring distance between the actual and the target
  values (`y`) if using supervised dimension reduction. Must be one of
  {"categorical", "euclidean"}. Default: "categorical".

- target_weight:

  Weighting factor between data topology and target topology. A value of
  0.0 weights entirely on data, a value of 1.0 weights entirely on
  target. The default of 0.5 balances the weighting equally between data
  and target.

- transform_input:

  If TRUE, then compute an approximate representation of the input data.
  Default: TRUE.

- seed:

  Optional seed for pseudo random number generator. Default: NULL.
  Setting a PRNG seed will enable consistency of trained embeddings,
  allowing for reproducible results to 3 digits of precision, but at the
  expense of potentially slower training and increased memory usage. If
  the PRNG seed is not set, then the trained embeddings will not be
  deterministic.

## Value

A UMAP model object that can be used as input to the
[`cuda_ml_transform()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_transform.md)
function. If `transform_input` is set to TRUE, then the model object
will contain a "transformed_data" attribute containing the lower
dimensional embedding of the input data.

## Examples

``` r
library(cuda.ml)

if (interactive() && cuda_ml_backend_info()$runtime_installed) {
  model <- cuda_ml_umap(
    x = iris[1:4],
    y = iris[[5]],
    n_components = 2,
    n_epochs = 200,
    transform_input = TRUE
  )

  set.seed(0L)
  print(kmeans(model$transformed_data, iter.max = 100, centers = 3))
}
```
