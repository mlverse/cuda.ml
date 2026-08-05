# Configure an approximate KNN query algorithm

For the main path, pass `"ivfflat"` or `"ivfpq"` directly to the `algo`
argument of
[`cuda_ml_knn()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_knn.md);
cuda.ml then lets the backend choose the index parameters. Use these
constructors only when those parameters need to be set explicitly.

## Usage

``` r
cuda_ml_knn_algo_ivfflat(nlist, nprobe)

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

A KNN algorithm specification to pass to the `algo` argument of
[`cuda_ml_knn()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_knn.md).

## Details

Both algorithms partition the training data into `nlist` cells and
search `nprobe` cells for each query. IVFFlat stores the original
vectors and therefore needs only those two parameters. IVFPQ also
compresses vectors using product quantization, so it additionally
requires the number of subquantizers (`m`) and the bits allocated to
each subquantizer (`n_bits`). The distinct constructors keep the
required parameters for each algorithm explicit.

## See also

[`cuda_ml_knn()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_knn.md)
