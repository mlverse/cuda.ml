# Audit the installed native backend

Recomputes the hashes recorded when the requested backend was installed
and validates its native registration. For the complete downloaded
backend, it also validates the managed-runtime dependency closure.
Ordinary runtime reuse performs only fast marker, inventory, size, and
link checks.

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
