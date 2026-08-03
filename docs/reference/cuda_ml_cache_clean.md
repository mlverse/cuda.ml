# Remove managed cuda.ml runtime caches

Removes cuda.ml runtime and backend cache generations. Restart R before
calling this function if the native backend has been loaded in this
process.

## Usage

``` r
cuda_ml_cache_clean()
```

## Value

Invisibly returns `TRUE`.
