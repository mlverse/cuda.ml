# Return individual-tree predictions

Return individual-tree predictions

## Usage

``` r
cuda_ml_nvforest_predict_per_tree(object, new_data, chunk_size = NULL)
```

## Arguments

- object:

  An nvForest-backed model.

- new_data:

  Numeric predictor data.

- chunk_size:

  Native prediction chunk size, or `NULL` for the model default. It
  controls native batching and does not limit the size of the returned R
  object.

## Value

For scalar-leaf models, a numeric matrix with one column per tree. For
vector-leaf models, a numeric array indexed by observation, tree, and
output.

## Memory use

The complete result is materialized in R: rows by trees for scalar-leaf
models and rows by trees by outputs for vector-leaf models. The
`chunk_size` argument controls native prediction work but does not bound
the memory required by the R result.

## See also

[`cuda_ml_nvforest_leaf_ids()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_leaf_ids.md)
