# Audit the installed managed runtime

Recomputes the hashes recorded when the managed runtime was installed
and validates the complete native dependency closure. Ordinary runtime
reuse performs only fast marker, inventory, size, and link checks.

## Usage

``` r
cuda_ml_runtime_audit()
```

## Value

Invisibly returns `TRUE`.
