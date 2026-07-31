# Make predictions on new data points.

Make predictions on new data points using a CuML SVM model.

## Usage

``` r
# S3 method for class 'cuda_ml_svm'
predict(object, new_data, ...)
```

## Arguments

- object:

  A trained CuML model.

- new_data:

  A matrix or data frame containing new data points.

- ...:

  Additional arguments to
  [`predict()`](https://rdrr.io/r/stats/predict.html). Currently unused.

## Value

Predictions on new data points.
