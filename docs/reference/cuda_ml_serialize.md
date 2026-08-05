# Save and restore supported cuda.ml models

`cuda_ml_serialize()` saves the explicit state of a fitted cuda.ml
model. `cuda_ml_unserialize()` restores that state as a fitted model.

## Usage

``` r
cuda_ml_serialize(model, connection = NULL, ...)

cuda_ml_unserialize(
  connection,
  ...,
  device = NULL,
  device_id = NULL,
  layout = NULL,
  precision = NULL,
  default_chunk_size = NULL,
  align_bytes = NULL
)
```

## Arguments

- model:

  The model object.

- connection:

  For `cuda_ml_serialize()`, a file path, an open connection, or `NULL`;
  a file path writes a gzip-compressed state and `NULL` returns the
  state as a raw vector. For `cuda_ml_unserialize()`, a file path, an
  open connection, or a raw vector.

- ...:

  Additional arguments passed to
  [`base::serialize()`](https://rdrr.io/r/base/serialize.html) or
  [`base::unserialize()`](https://rdrr.io/r/base/serialize.html).

- device, device_id, layout, precision, default_chunk_size, align_bytes:

  Named nvForest inference options. They are supported only for nvForest
  and random-forest states. When `device` is omitted, those states
  restore for GPU inference. When `precision` is omitted, the saved
  prediction precision is used. The remaining omitted options use
  nvForest defaults.

## Value

`cuda_ml_serialize()` returns `NULL` when writing to a file or
connection and otherwise returns a raw vector. `cuda_ml_unserialize()`
returns the restored fitted model.

## Supported models

Explicit state is supported for:

- OLS, ridge, lasso, elastic-net, and SGD linear models;

- logistic and multinomial regression;

- PCA;

- binary and one-vs-rest SVC models and SVR models;

- UMAP;

- random forests and other nvForest-backed models.

KNN and TSVD fits are not currently supported. The pinned KNN API does
not expose portable approximate-index state, and the current TSVD
binding retains native transform parameters that cuda.ml does not
reconstruct.

## Deployment

cuda.ml validates the model state and required backend before loading
it. Prepare the backend in the target process with
[`cuda_ml_install()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_install.md)
for GPU operation or `cuda_ml_install(device = "cpu")` for CPU-only
nvForest inference.

Random-forest and nvForest states contain device-neutral Treelite model
bytes. They retain prediction precision, class labels, preprocessing,
and model semantics, but not the inference device, device identifier,
tree layout, chunk size, or memory alignment. Select those settings
while restoring; omitting `device` selects GPU inference.

[`bundle::bundle()`](https://rstudio.github.io/bundle/reference/bundle.html)
stores the same explicit state. For an nvForest-backed model, the bundle
also stores its chosen deployment device separately from the
device-neutral state. A bundle is not required for deployment;
`cuda_ml_serialize()` returns the complete state artifact directly.

## See also

[`serialize`](https://rdrr.io/r/base/serialize.html),
[`unserialize`](https://rdrr.io/r/base/serialize.html), and
[`bundle`](https://rstudio.github.io/bundle/reference/bundle.html)
