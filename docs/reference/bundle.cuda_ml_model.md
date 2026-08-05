# Bundle a cuda.ml model

Converts a model with explicit state into a
[`bundle::bundle()`](https://rstudio.github.io/bundle/reference/bundle.html)
object. KNN and TSVD fits do not currently implement explicit state.

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
[`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
returns the complete state artifact directly.

## See also

[`cuda_ml_serialize`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md),
[`cuda_ml_unserialize`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
