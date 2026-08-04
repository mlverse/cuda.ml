# Make predictions on new data points.

Make predictions on new data points using a cuML KNN model.

## Usage

``` r
# S3 method for class 'cuda_ml_knn'
predict(object, new_data, type = NULL, ...)
```

## Arguments

- object:

  A trained CuML model.

- new_data:

  A matrix or data frame containing new data points.

- type:

  Type of prediction. Classification models support `"class"` and
  `"prob"`; regression models support `"numeric"`. The default is
  `"class"` for classification and `"numeric"` for regression.

- ...:

  Additional arguments to
  [`predict()`](https://rdrr.io/r/stats/predict.html). Currently unused.

## Value

Predictions on new data points.
