# Report backend and managed-runtime metadata

Report backend and managed-runtime metadata

## Usage

``` r
cuda_ml_backend_info()
```

## Value

A named list describing the packaged backend and whether its exact
managed-runtime cache is complete. This function performs read-only
cache and inventory checks. It does not create or modify the cache,
access the network, inspect an NVIDIA GPU or driver, or load the native
backend.
