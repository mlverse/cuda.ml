# Save and restore models

Some fitted cuda.ml objects contain native pointers that are meaningful
only in the R process that created them. Use an explicit cuda.ml model
state, a `bundle` object, or an nvForest checkpoint pair instead of
relying on [`saveRDS()`](https://rdrr.io/r/base/readRDS.html) to capture
a live fitted object.

## Choose a persistence format

| Need                                         | Save                                                                                                                                | Restore                                                                                                                               | Result                                     |
|:---------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------|
| One R-native artifact                        | [`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)                                           | [`cuda_ml_unserialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)                                           | A cuda.ml model state                      |
| Integration with the `bundle` package        | [`bundle::bundle()`](https://rstudio.github.io/bundle/reference/bundle.html) and [`saveRDS()`](https://rdrr.io/r/base/readRDS.html) | [`readRDS()`](https://rdrr.io/r/base/readRDS.html) and [`bundle::unbundle()`](https://rstudio.github.io/bundle/reference/bundle.html) | A bundle containing the same cuda.ml state |
| A Treelite checkpoint usable outside cuda.ml | [`cuda_ml_nvforest_export()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md)                               | [`cuda_ml_nvforest_import()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md)                                 | A checkpoint plus cuda.ml metadata         |

The first two choices support cuda.ml models that implement explicit
model state. The checkpoint choice is only for random forests and other
nvForest-backed models. It is described in more detail in the [nvForest
inference
guide](https://mlverse.github.io/cuda.ml/articles/nvforest.md).

## Save directly to a file

The simplest file workflow passes a path directly. cuda.ml writes a
gzip-compressed model state rather than a live native pointer.

``` r
library(cuda.ml)

cuda_ml_install()

model <- cuda_ml_linear_reg(
  mpg ~ .,
  data = mtcars,
  penalty = 0.01,
  mixture = 0
)

state_path <- tempfile(fileext = ".cuda-ml-state")
cuda_ml_serialize(model, state_path)
#> NULL
```

In a new R process, prepare the required backend and restore the model.
cuda.ml validates the state and backend before loading it.

``` r
library(cuda.ml)

cuda_ml_install()

model <- cuda_ml_unserialize(state_path)

predict(model, mtcars[1:5, names(mtcars) != "mpg"])
#> # A tibble: 5 × 1
#>   .pred
#>   <dbl>
#> 1  22.6
#> 2  22.1
#> 3  26.3
#> 4  21.2
#> 5  17.7
```

## Keep the state as raw bytes

With its default `connection = NULL`,
[`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
returns the uncompressed state as a raw vector. This is useful for
object stores and other systems that accept bytes directly.

``` r
state <- cuda_ml_serialize(model)
str(state)
#>  raw [1:4327] 58 0a 00 00 ...

model <- cuda_ml_unserialize(state)
```

The `blob` package can wrap this raw vector as one database BLOB value.

## Use a bundle

The `bundle` package wraps the same explicit cuda.ml state and records
how to restore it. This is useful in workflows that already use
`bundle`.

``` r
library(bundle)

bundle_path <- tempfile(fileext = ".bundle.rds")
bundled_model <- bundle(model)
saveRDS(bundled_model, bundle_path)

bundled_model <- readRDS(bundle_path)
model <- unbundle(bundled_model)
```

For an nvForest-backed model, `device` chooses where the bundle will
restore. Use this when training a random forest on a GPU and deploying
it on a CPU-only host.

``` r
forest <- cuda_ml_rand_forest(
  Species ~ .,
  data = iris,
  trees = 100,
  seed = 1
)

cpu_bundle <- bundle(forest, device = "cpu")
forest_bundle_path <- tempfile(fileext = ".bundle.rds")
saveRDS(cpu_bundle, forest_bundle_path)
```

The target host must prepare a backend that supports CPU inference
before calling
[`unbundle()`](https://rstudio.github.io/bundle/reference/bundle.html).
For a smaller CPU-only deployment:

``` r
library(cuda.ml)
library(bundle)

cuda_ml_install(device = "cpu")
forest <- unbundle(readRDS(forest_bundle_path))
```

The `device` argument is supported only for nvForest-backed models.

## Export an nvForest checkpoint pair

[`cuda_ml_nvforest_export()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md)
writes two files:

- `<prefix>.treelite.checkpoint`, containing the device-neutral trees;
- `<prefix>.cuda-ml.json`, containing the metadata needed for an exact
  cuda.ml round-trip.

``` r
forest_directory <- tempfile("forest-artifact-")
dir.create(forest_directory)
cuda_ml_nvforest_export(
  forest,
  directory = forest_directory,
  prefix = "model"
)
```

Copy both files when another cuda.ml process will restore the model.
Select the deployment device during import:

``` r
cuda_ml_install(device = "cpu")

forest <- cuda_ml_nvforest_import(
  directory = forest_directory,
  prefix = "model",
  device = "cpu"
)
```

Other Treelite consumers can read the checkpoint alone, but they must
supply predictors in the recorded order and reproduce any class-label
and postprocessing semantics in the JSON sidecar. Loading the bare
checkpoint back into cuda.ml does not recover those semantics. Use the
pair for an exact round-trip.

## Select the nvForest restore device

Current random-forest and nvForest states contain device-neutral
Treelite model bytes. Select CPU or GPU inference while restoring:

``` r
forest_state <- cuda_ml_serialize(forest)
cpu_forest <- cuda_ml_unserialize(forest_state, device = "cpu")
gpu_forest <- cuda_ml_unserialize(
  forest_state,
  device = "gpu",
  device_id = 0
)
```

If `device` is omitted, these states restore for GPU inference. The
state keeps prediction precision, class labels, preprocessing, and model
semantics. It does not keep the deployment device, device identifier,
tree layout, chunk size, or memory alignment. Restore-time inference
options are supported only for current nvForest and random-forest
states.

Install the complete backend for GPU operation or the smaller CPU
backend for CPU-only nvForest inference. See the [installation and
runtime
guide](https://mlverse.github.io/cuda.ml/articles/install-manage.md) for
those workflows.

## Treat model artifacts as trusted input

Load cuda.ml states, bundles, and checkpoint sidecars only from trusted
sources. An nvForest JSON sidecar embeds an R-serialized preprocessing
blueprint. Its SHA-256 digest checks checkpoint integrity, not the
identity of the artifact’s producer.
