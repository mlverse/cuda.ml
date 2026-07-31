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

  Prediction chunk size, or `NULL` for the model default.

## Value

For scalar-leaf models, a numeric matrix with one column per tree. For
vector-leaf models, a numeric array indexed by observation, tree, and
output.
