# Bundle a cuda.ml model

Converts a model with explicit state into a
[`bundle::bundle()`](https://rstudio.github.io/bundle/reference/bundle.html)
object. Models without an explicit state fail rather than serializing
native pointers.

## Usage

``` r
# S3 method for class 'cuda_ml_model'
bundle(x, ...)

# S3 method for class 'cuda_ml_nvforest'
bundle(x, device = NULL, ...)
```

## Arguments

- x:

  A fitted cuda.ml model.

- ...:

  Unused.

- device:

  For an nvForest-backed model, the device on which the bundle will
  restore. `NULL` preserves the model's current device. Use `"cpu"` when
  bundling a GPU-trained random forest for CPU-only deployment. Other
  cuda.ml model types do not accept this argument.

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
[`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
returns the complete state artifact directly.

A restored fit supports the same prediction or transformation operations
as the original fit. cuda.ml does not provide warm-start,
incremental-training, or fine-tuning operations for either live or
restored fits. Refit a model by calling its fitting function again with
training data.

## See also

[`cuda_ml_serialize`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md),
[`cuda_ml_unserialize`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
