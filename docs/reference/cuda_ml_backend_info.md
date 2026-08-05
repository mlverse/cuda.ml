# Report native-backend metadata

Performs cheap, read-only cache and inventory checks for routine
inspection. It reports whether the selected backend cache is complete,
but does not verify every recorded hash or native registration. It does
not create or modify the cache, access the network, inspect an NVIDIA
GPU or driver, or load native code.

## Usage

``` r
cuda_ml_backend_info()
```

## Value

A named list describing the selected backend, pinned library versions,
cache status, and CPU-only nvForest backend status.

## Details

Use
[`cuda_ml_runtime_audit()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_runtime_audit.md)
when an explicit deep integrity check is needed. The audit recomputes
recorded hashes, validates native registration, and, for the complete
downloaded backend, validates the managed-runtime dependency closure.

## See also

[`cuda_ml_runtime_audit()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_runtime_audit.md)
