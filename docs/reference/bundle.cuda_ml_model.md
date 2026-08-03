# Bundle a cuda.ml model

Converts a model with an explicit portable state into a
[`bundle::bundle()`](https://rstudio.github.io/bundle/reference/bundle.html)
object. Models without an explicit state fail rather than serializing
native pointers.

## Usage

``` r
# S3 method for class 'cuda_ml_model'
bundle(x, ...)
```

## Arguments

- x:

  A fitted cuda.ml model.

- ...:

  Unused.
