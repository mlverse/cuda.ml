# Build a specification for the "ivfflat" KNN query algorithm.

Build a specification of the flat-inverted-file KNN query algorithm,
with all required parameters specified explicitly.

## Usage

``` r
cuda_ml_knn_algo_ivfflat(nlist, nprobe)
```

## Arguments

- nlist:

  Number of cells to partition dataset into.

- nprobe:

  At query time, the number of cells used for approximate nearest
  neighbor search.

## Value

An object encapsulating all required parameters of the "ivfflat" KNN
query algorithm.
