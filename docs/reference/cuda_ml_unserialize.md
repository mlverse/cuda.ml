# Unserialize a CuML model state

Unserialize a CuML model state into a CuML model object.

## Usage

``` r
cuda_ml_unserialize(connection, ...)
```

## Arguments

- connection:

  An open connection or a raw vector.

- ...:

  Additional arguments to
  [`base::unserialize()`](https://rdrr.io/r/base/serialize.html).

## Value

A unserialized CuML model.

## See also

[`unserialize`](https://rdrr.io/r/base/serialize.html)
