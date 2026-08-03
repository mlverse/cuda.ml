# Predict with an nvForest model

Predict with an nvForest model

## Usage

``` r
# S3 method for class 'cuda_ml_nvforest'
predict(
  object,
  new_data,
  type = NULL,
  threshold = NULL,
  chunk_size = NULL,
  ...
)
```

## Arguments

- object:

  An nvForest-backed model.

- new_data:

  Numeric predictor data.

- type:

  Classification models support `"class"` and `"prob"`; regression
  models support `"numeric"`.

- threshold:

  Binary classification threshold, or `NULL` for 0.5.

- chunk_size:

  Prediction chunk size, or `NULL` for the model default.

- ...:

  Unused.
