# Build a specification for the "ivfpq" KNN query algorithm.

Build a specification of the inverted-file-product-quantization KNN
query algorithm, with all required parameters specified explicitly.

## Usage

``` r
cuda_ml_knn_algo_ivfpq(nlist, nprobe, m, n_bits)
```

## Arguments

- nlist:

  Number of cells to partition dataset into.

- nprobe:

  At query time, the number of cells used for approximate nearest
  neighbor search.

- m:

  Number of subquantizers.

- n_bits:

  Bits allocated per subquantizer, from 4 to 8. The product of `m` and
  `n_bits` must be divisible by 8.

## Value

An object encapsulating all required parameters of the "ivfpq" KNN query
algorithm.
