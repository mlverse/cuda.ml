# Remove cuda.ml native-backend caches

Removes downloaded and source-built runtime and backend cache
generations, including the selected-backend record. Restart R before
calling this function if the native backend has been loaded in this
process.

## Usage

``` r
cuda_ml_cache_clean()
```

## Value

Invisibly returns `TRUE`.
