# Predict from a logistic or multinomial regression model

Predict from a logistic or multinomial regression model

## Usage

``` r
# S3 method for class 'cuda_ml_logistic_reg'
predict(object, new_data, type = c("class", "prob"), ...)
```

## Arguments

- object:

  A fitted `cuda_ml_logistic_reg` model.

- new_data:

  New predictor data.

- type:

  Either `"class"` or `"prob"`.

- ...:

  Unused.
