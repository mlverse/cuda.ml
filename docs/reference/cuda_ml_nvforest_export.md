# Export and import an nvForest checkpoint pair

`cuda_ml_nvforest_export()` writes a standard Treelite checkpoint and a
cuda.ml JSON sidecar. The checkpoint contains the device-neutral tree
ensemble. The sidecar retains cuda.ml compatibility metadata, class
labels, prediction precision, random-forest probability semantics, and
the R preprocessing blueprint needed for a complete cuda.ml round-trip.
`cuda_ml_nvforest_import()` restores the pair on a caller-selected
inference device.

## Usage

``` r
cuda_ml_nvforest_export(object, directory, prefix, overwrite = FALSE)

cuda_ml_nvforest_import(
  directory,
  prefix,
  device = c("gpu", "cpu"),
  device_id = NULL,
  layout = c("depth_first", "breadth_first", "layered"),
  precision = NULL,
  default_chunk_size = NULL,
  align_bytes = NULL
)
```

## Arguments

- object:

  An nvForest-backed model.

- directory:

  An existing output directory.

- prefix:

  A non-empty filename prefix without directory components.

- overwrite:

  Whether to replace both existing output files. The default is `FALSE`.

- device:

  Inference device: `"gpu"` or `"cpu"`. The default is `"gpu"`.

- device_id:

  GPU device identifier, or `NULL` for the current device.

- layout:

  Tree layout.

- precision:

  Native, single, or double precision. `NULL` retains the exported
  model's prediction precision.

- default_chunk_size:

  Default prediction chunk size, or `NULL` to use nvForest's heuristic.

- align_bytes:

  Memory alignment, or `NULL` for the device default.

## Value

`cuda_ml_nvforest_export()` invisibly returns a named character vector
containing the absolute `checkpoint` and `metadata` paths.
`cuda_ml_nvforest_import()` returns the restored nvForest-backed model.

## Files

The function writes exactly `<prefix>.treelite.checkpoint` and
`<prefix>.cuda-ml.json`. The JSON records the checkpoint's relative
filename, size, and SHA-256 digest. It does not record inference device,
layout, chunk size, memory alignment, or GPU device identifier.

Other Treelite consumers can load the checkpoint without the JSON. They
must supply numeric predictors in the recorded processed feature order
when feature names are available, or in the checkpoint's original
positional order otherwise. They must also implement any class-label and
postprocessing behavior described by the sidecar.

Loading the bare checkpoint with
`cuda_ml_nvforest_load_model(model_type = "treelite_checkpoint")`
likewise omits the sidecar's preprocessing, original class labels,
cuda.ml model class, and random-forest probability semantics. Use
`cuda_ml_nvforest_import()` for an exact cuda.ml round-trip.

## Persistence choices

Use
[`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
and
[`cuda_ml_unserialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
for one R-native state value. The checkpoint pair is useful when the
Treelite model must also be independently available. A bundle is
optional wrapping around the R-native state and is not required for
either workflow.

Import requires the exact Treelite version recorded by the sidecar.
Prepare the selected backend before import:
[`cuda_ml_install()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_install.md)
for GPU operation or `cuda_ml_install(device = "cpu")` for CPU-only
inference. Import never downloads a backend.

## Trust

The JSON embeds an R-serialized hardhat blueprint so that formula and
recipe preprocessing round-trip. Import only artifacts from trusted
sources, as with [`readRDS()`](https://rdrr.io/r/base/readRDS.html) and
[`cuda_ml_unserialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md).
The recorded SHA-256 digest checks integrity, not authenticity.

## See also

[`cuda_ml_nvforest_load_model()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_load_model.md)
and
[`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
