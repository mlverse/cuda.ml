# Perform t-distributed stochastic neighbor embedding.

t-distributed stochastic neighbor embedding (t-SNE) for visualizing
high- dimensional data.

## Usage

``` r
cuda_ml_tsne(
  x,
  n_components = 2L,
  n_neighbors = ceiling(3 * perplexity),
  method = c("barnes_hut", "fft", "exact"),
  angle = 0.5,
  n_iter = 1000L,
  learning_rate = 200,
  learning_rate_method = c("adaptive", "none"),
  perplexity = 30,
  perplexity_max_iter = 100L,
  perplexity_tol = 1e-05,
  early_exaggeration = 12,
  late_exaggeration = 1,
  exaggeration_iter = 250L,
  min_grad_norm = 1e-07,
  pre_momentum = 0.5,
  post_momentum = 0.8,
  square_distances = TRUE,
  seed = NULL
)
```

## Arguments

- x:

  The input matrix or data frame. Each data point should be a row and
  should consist of numeric values only.

- n_components:

  Dimension of the embedded space.

- n_neighbors:

  The number of datapoints to use in the attractive forces. Default:
  ceiling(3 \* perplexity).

- method:

  T-SNE method, must be one of {"barnes_hut", "fft", "exact"}. The
  "exact" method will be more accurate but slower. Both "barnes_hut" and
  "fft" methods are fast approximations.

- angle:

  Valid values are between 0.0 and 1.0, which trade off speed and
  accuracy, respectively. Generally, these values are set between 0.2
  and 0.8. (Barnes-Hut only.)

- n_iter:

  Maximum number of iterations for the optimization. Should be at
  least 250. Default: 1000L.

- learning_rate:

  Learning rate of the t-SNE algorithm, usually between (10, 1000). If
  the learning rate is too high, then t-SNE result could look like a
  cloud / ball of points.

- learning_rate_method:

  Must be one of {"adaptive", "none"}. If "adaptive", then learning
  rate, early exaggeration, and perplexity are automatically tuned based
  on input size. Default: "adaptive".

- perplexity:

  The target value of the conditional distribution's perplexity (see
  https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
  for details).

- perplexity_max_iter:

  The number of epochs the best Gaussian bands are found for. Default:
  100L.

- perplexity_tol:

  Stop optimizing the Gaussian bands when the conditional distribution's
  perplexity is within this desired tolerance compared to its target
  value. Default: 1e-5.

- early_exaggeration:

  Controls the space between clusters. Not critical to tune this.
  Default: 12.0.

- late_exaggeration:

  Controls the space between clusters. It may be beneficial to increase
  this slightly to improve cluster separation. This will be applied
  after `exaggeration_iter` iterations (FFT only).

- exaggeration_iter:

  Number of exaggeration iterations. Default: 250L.

- min_grad_norm:

  If the gradient norm is below this threshold, the optimization will be
  stopped. Default: 1e-7.

- pre_momentum:

  During the exaggeration iteration, more forcefully apply gradients.
  Default: 0.5.

- post_momentum:

  During the late phases, less forcefully apply gradients. Default: 0.8.

- square_distances:

  Whether TSNE should square the distance values.

- seed:

  Seed to the pseudorandom number generator. Setting this can make
  repeated runs look more similar. Note, however, that this highly
  parallelized t-SNE implementation is not completely deterministic
  between runs, even with the same `seed` being used for each run.
  Default: NULL.

## Value

A matrix containing the embedding of the input data in a low-
dimensional space, with each row representing an embedded data point.

## Examples

``` r
library(cuda.ml)

if (interactive() && cuda_ml_backend_info()$runtime_installed) {
  oil_predictors <- scale(
    modeldata::oils[names(modeldata::oils) != "class"]
  )

  embedding <- cuda_ml_tsne(oil_predictors, method = "exact")

  set.seed(0L)
  print(kmeans(embedding, centers = 7))
}
```
