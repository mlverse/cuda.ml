# Audit the installed native backend

Performs an explicit deep integrity check. It recomputes the hashes
recorded when the requested backend was installed and validates native
registration, loading the backend temporarily when needed. For the
complete downloaded backend, it also validates the managed-runtime
dependency closure.

## Usage

``` r
cuda_ml_runtime_audit(device = c("gpu", "cpu"))
```

## Arguments

- device:

  Backend to audit: the complete `"gpu"` backend or the CPU-only
  nvForest backend.

## Value

Invisibly returns `TRUE`.

## Details

Use
[`cuda_ml_backend_info()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_backend_info.md)
for routine, read-only inspection. That function performs only fast
cache and inventory checks and does not load native code. Ordinary
runtime reuse likewise performs only fast marker, inventory, size, and
link checks.

## See also

[`cuda_ml_backend_info()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_backend_info.md)
