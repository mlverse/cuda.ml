# Serialize a CuML model

Given a CuML model, serialize its state into a connection.

## Usage

``` r
cuda_ml_serialize(model, connection = NULL, ...)
```

## Arguments

- model:

  The model object.

- connection:

  An open connection or `NULL`. If `NULL`, then the model state is
  serialized to a raw vector. Default: NULL.

- ...:

  Additional arguments to
  [`base::serialize()`](https://rdrr.io/r/base/serialize.html).

## Value

`NULL` unless `connection` is `NULL`, in which case the serialized model
state is returned as a raw vector.

## See also

[`serialize`](https://rdrr.io/r/base/serialize.html)
