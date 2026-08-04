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
  models support `"numeric"`. Probability prediction is available only
  when `cuda_ml_nvforest_info(object)$has_probability_output` is true.
  Unsupported Treelite postprocessors fail explicitly.

- threshold:

  Binary classification threshold, or `NULL` for 0.5.

- chunk_size:

  Native prediction chunk size, or `NULL` for the model default. It
  controls native batching and does not limit the size of the returned R
  object.

- ...:

  Unused.

## Value

A tibble with `.pred` for regression, `.pred_class` for class
prediction, or one probability column named `.pred_<level>` for each
class.
