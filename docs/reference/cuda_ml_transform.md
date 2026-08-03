# Transform data using a trained cuML model.

Given a trained cuML model, transform an input dataset using that model.

## Usage

``` r
cuda_ml_transform(model, x, ...)
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
