# Apply the inverse transformation defined by a trained cuML model.

Given a trained cuML model, apply the inverse transformation defined by
that model to an input dataset.

## Usage

``` r
cuda_ml_inverse_transform(model, x, ...)
```

## Arguments

- model:

  A model object.

- x:

  The dataset to be transformed.

- ...:

  Additional model-specific parameters (if any).

## Value

The transformed data points.
