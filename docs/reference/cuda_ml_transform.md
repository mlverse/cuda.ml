# Transform data with a dimensionality-reduction model

These generics apply a fitted dimensionality-reduction mapping. They are
distinct from [`predict()`](https://rdrr.io/r/stats/predict.html), which
produces outcomes from supervised models and returns tidymodels-style
prediction columns.

## Usage

``` r
cuda_ml_transform(model, x, ...)

cuda_ml_inverse_transform(model, x, ...)
```

## Arguments

- model:

  A model object.

- x:

  The dataset to be transformed.

- ...:

  Additional model-specific parameters (if any).

## Value

`cuda_ml_transform()` returns coordinates in the learned representation.
`cuda_ml_inverse_transform()` returns reconstructed predictors in the
original feature space.

## Supported methods

- `cuda_ml_transform()` maps predictors into a learned lower-dimensional
  representation. It supports fitted TSVD and UMAP models.

- `cuda_ml_inverse_transform()` maps component coordinates back toward
  the original feature space. It supports fitted PCA and TSVD models.

PCA stores the transformed training input when `transform_input = TRUE`,
but it does not currently provide a method for transforming new data.

## See also

[`predict`](https://rdrr.io/r/stats/predict.html),
[`cuda_ml_pca`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_pca.md),
[`cuda_ml_tsvd`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_tsvd.md),
and
[`cuda_ml_umap`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_umap.md)
