# Return terminal leaf identifiers

Return terminal leaf identifiers

## Usage

``` r
cuda_ml_nvforest_leaf_ids(object, new_data, chunk_size = NULL)
```

## Arguments

- object:

  An nvForest-backed model.

- new_data:

  Numeric predictor data.

- chunk_size:

  Prediction chunk size, or `NULL` for the model default.

## Value

An integer matrix with one row per observation and one column per tree.
