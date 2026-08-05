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

  For `cuda_ml_serialize()`, an open connection or `NULL`; `NULL`
  returns the state as a raw vector. For `cuda_ml_unserialize()`, an
  open connection or a raw vector.

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

`cuda_ml_serialize()` returns `NULL` when writing to a connection and
otherwise returns a raw vector. `cuda_ml_unserialize()` returns the
restored fitted model.

## Supported models

Explicit state is supported for:

- OLS, ridge, lasso, elastic-net, and SGD linear models;

- logistic and multinomial regression;

- PCA;

- binary and one-vs-rest SVC models and SVR models;

- UMAP;

- random forests and other nvForest-backed models.

Other cuda.ml models fail during `cuda_ml_serialize()` instead of saving
native pointers that cannot be used in another R process.

## Compatibility

The cuda.ml package version that created a state is recorded as
provenance; a different package version does not by itself prevent
restoration. Backend compatibility depends on the model family:

- Linear and logistic-regression states do not require a matching
  backend version.

- PCA, SVC, one-vs-rest SVC, SVR, and UMAP states require the same
  RAPIDS version.

- Random-forest and nvForest states require the same Treelite version.

Other recorded backend details are provenance and do not gate
restoration. The package checks the saved model type and required
metadata before loading its payload, and rejects unsupported or
incompatible states.

Random-forest and nvForest states contain device-neutral Treelite model
bytes. They retain prediction precision, class labels, preprocessing,
and model semantics, but not the inference device, device identifier,
tree layout, chunk size, or memory alignment. Select those settings
while restoring; omitting `device` selects GPU inference.

Saving a state to a file connection and restoring it in another R
process uses this same contract. The target process must have a
compatible cuda.ml installation and must prepare the corresponding
backend before prediction:
[`cuda_ml_install()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_install.md)
for GPU operation or `cuda_ml_install(device = "cpu")` for CPU-only
nvForest inference.
[`bundle::bundle()`](https://rstudio.github.io/bundle/reference/bundle.html)
stores the same explicit state, so saving a bundle with
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html) and restoring it with
[`readRDS()`](https://rdrr.io/r/base/readRDS.html) and
[`bundle::unbundle()`](https://rstudio.github.io/bundle/reference/bundle.html)
has the same compatibility requirements. For an nvForest-backed model,
the bundle also stores its chosen deployment device separately from the
device-neutral state. A bundle is not required for deployment;
`cuda_ml_serialize()` returns the complete state artifact directly.

A restored fit supports the same prediction or transformation operations
as the original fit. cuda.ml does not provide warm-start,
incremental-training, or fine-tuning operations for either live or
restored fits. Refit a model by calling its fitting function again with
training data.

## See also

[`serialize`](https://rdrr.io/r/base/serialize.html),
[`unserialize`](https://rdrr.io/r/base/serialize.html), and
[`bundle`](https://rstudio.github.io/bundle/reference/bundle.html)
