# Prepare the managed CUDA and RAPIDS runtime

Downloads, verifies, extracts, and caches the runtime libraries required
by the precompiled cuda.ml backend. This operation does not load the
backend or require a GPU, driver, CUDA toolkit, compiler, Python, or
conda. Calling it again with the same package build is a no-op.

## Usage

``` r
cuda_ml_install()
```

## Value

Invisibly returns `TRUE`.

## Details

The default cache is `tools::R_user_dir("cuda.ml", "cache")`. Set
`CUDA_ML_CACHE_DIR` to use a different cache root.

## Examples

``` r
if (FALSE) { # \dontrun{
cuda_ml_install()
} # }
```
