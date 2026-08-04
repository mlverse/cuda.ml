# Unserialize a cuML model state

Unserialize a cuML model state into a cuML model object.

## Usage

``` r
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

- connection:

  An open connection or a raw vector.

- ...:

  Additional arguments to
  [`base::unserialize()`](https://rdrr.io/r/base/serialize.html).

- device, device_id, layout, precision, default_chunk_size, align_bytes:

  Named nvForest inference options. They are supported only for
  device-neutral nvForest and random-forest states. When `device` is
  omitted, these states restore for GPU inference. When `precision` is
  omitted, the precision recorded in the state is used. The remaining
  omitted options use nvForest defaults.

## Value

An unserialized cuML model.

## Persistence contract

cuda.ml schema 1 model states contain a schema number, the cuda.ml
package version that created the state, backend provenance, a model ABI
identifier, and a payload. The package version is provenance only: a
difference from the installed cuda.ml version does not prevent
restoration.

Compatibility is determined before the payload is restored:

- The schema must be the integer `1`. Unknown and unversioned schemas
  are rejected.

- The state class and model ABI must identify a restoration method
  supported by the installed package. A change to a model's payload
  layout requires a new model ABI.

- Linear-model and logistic-regression states have portable R payloads
  and do not require matching backend identity fields.

- PCA, SVC, one-vs-rest SVC, SVR, and UMAP states require an exact
  `rapids_version` match because their payloads reconstruct RAPIDS
  native state.

- Random-forest and nvForest states require an exact `treelite_version`
  match because their payloads contain serialized Treelite model bytes.

The remaining recorded backend fields—`cuda_version`,
`nvforest_version`, and `platform`—are provenance for schema 1, not
compatibility gates. A missing payload, unsupported ABI, or missing or
unequal required backend field is rejected. cuda.ml does not implicitly
migrate a state or fall back to serializing native pointers.

Current nvForest and random-forest states store device-neutral Treelite
model bytes. They retain model semantics and prediction precision, but
not the inference device, device identifier, tree layout, chunk size, or
memory alignment. Select those settings when restoring with
`cuda_ml_unserialize()`; GPU is the default. Legacy v1 nvForest and
random-forest states remain supported and restore with the inference
settings recorded in their payloads.

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

## See also

[`unserialize`](https://rdrr.io/r/base/serialize.html)
