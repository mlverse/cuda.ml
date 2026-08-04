# Save and restore models

Some fitted cuda.ml objects contain native pointers that are meaningful
only in the R process that created them. Use an explicit cuda.ml model
state, a `bundle` object, or an nvForest checkpoint pair instead of
relying on [`saveRDS()`](https://rdrr.io/r/base/readRDS.html) to capture
a live fitted object.

The examples are not evaluated when this vignette is built. Building the
vignette therefore requires no GPU, native backend, runtime download, or
network access.

## Choose a persistence format

| Need                                         | Save                                                                                                                                | Restore                                                                                                                               | Result                                     |
|:---------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------|
| One R-native artifact                        | [`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)                                           | [`cuda_ml_unserialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_unserialize.md)                                         | A versioned cuda.ml model state            |
| Integration with the `bundle` package        | [`bundle::bundle()`](https://rstudio.github.io/bundle/reference/bundle.html) and [`saveRDS()`](https://rdrr.io/r/base/readRDS.html) | [`readRDS()`](https://rdrr.io/r/base/readRDS.html) and [`bundle::unbundle()`](https://rstudio.github.io/bundle/reference/bundle.html) | A bundle containing the same cuda.ml state |
| A Treelite checkpoint usable outside cuda.ml | [`cuda_ml_nvforest_export()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md)                               | [`cuda_ml_nvforest_import()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md)                                 | A checkpoint plus cuda.ml metadata         |

The first two choices support cuda.ml models that implement explicit
model state. The checkpoint choice is only for random forests and other
nvForest-backed models. It is described in more detail in the [nvForest
inference
guide](https://mlverse.github.io/cuda.ml/articles/nvforest.md).

## Save one R-native state

[`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
returns a raw vector when `connection = NULL`, its default. The vector
contains a versioned model state rather than a live native pointer.

``` r
library(cuda.ml)

cuda_ml_install()

model <- cuda_ml_linear_reg(
  mpg ~ .,
  data = mtcars,
  penalty = 0.01,
  mixture = 0
)

state <- cuda_ml_serialize(model)
saveRDS(state, "mtcars-ridge.cuda-ml-state.rds")
```

In a new R process, prepare the required backend before using the
restored model, then pass the saved raw vector to
[`cuda_ml_unserialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_unserialize.md).

``` r
library(cuda.ml)

cuda_ml_install()

state <- readRDS("mtcars-ridge.cuda-ml-state.rds")
model <- cuda_ml_unserialize(state)
predict(model, mtcars[1:5, names(mtcars) != "mpg"])
```

You can write the state directly to a binary connection instead:

``` r
connection <- file("mtcars-ridge.cuda-ml-state", open = "wb")
cuda_ml_serialize(model, connection)
close(connection)

connection <- file("mtcars-ridge.cuda-ml-state", open = "rb")
model <- cuda_ml_unserialize(connection)
close(connection)
```

## Use a bundle

The `bundle` package wraps the same explicit cuda.ml state and records
how to restore it. This is useful in workflows that already use
`bundle`; it does not change cuda.ml’s compatibility requirements.

``` r
library(bundle)

bundled_model <- bundle(model)
saveRDS(bundled_model, "mtcars-ridge.bundle.rds")

bundled_model <- readRDS("mtcars-ridge.bundle.rds")
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
saveRDS(cpu_bundle, "forest-cpu.bundle.rds")
```

The target host must install the CPU inference backend before calling
[`unbundle()`](https://rstudio.github.io/bundle/reference/bundle.html):

``` r
library(cuda.ml)
library(bundle)

cuda_ml_install(device = "cpu")
forest <- unbundle(readRDS("forest-cpu.bundle.rds"))
```

The `device` argument is supported only for nvForest-backed models.

## Export an nvForest checkpoint pair

[`cuda_ml_nvforest_export()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md)
writes two files:

- `<prefix>.treelite.checkpoint`, containing the device-neutral trees;
- `<prefix>.cuda-ml.json`, containing the metadata needed for an exact
  cuda.ml round-trip.

``` r
dir.create("forest-artifact")
cuda_ml_nvforest_export(
  forest,
  directory = "forest-artifact",
  prefix = "model"
)
```

Copy both files when another cuda.ml process will restore the model.
Select the deployment device during import:

``` r
cuda_ml_install(device = "cpu")

forest <- cuda_ml_nvforest_import(
  directory = "forest-artifact",
  prefix = "model",
  device = "cpu"
)
```

Other Treelite consumers can read the checkpoint alone, but they must
supply predictors in the recorded order and reproduce any class-label
and postprocessing semantics in the JSON sidecar. Loading the bare
checkpoint back into cuda.ml does not recover those semantics. Use the
pair for an exact round-trip.

## Compatibility rules

Every current cuda.ml state uses schema 1 and records its model ABI,
package version, backend provenance, and payload. The package version is
provenance; a different package version does not by itself prevent
restoration. Compatibility depends on the state schema, model ABI, and
the native format used by the payload.

| Model state                              | Required backend identity                                       |
|:-----------------------------------------|:----------------------------------------------------------------|
| Linear and logistic regression           | No backend identity match; the state payload is portable R data |
| PCA, SVC, one-vs-rest SVC, SVR, and UMAP | Exact RAPIDS version                                            |
| Random forest and nvForest               | Exact Treelite version                                          |

These model families implement explicit state in the current release.
Models without explicit state support fail during
[`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
rather than falling back to serialization of native pointers.

States are not migrated implicitly. An unknown schema, unsupported model
ABI, missing payload, or missing or unequal required backend field
produces an error before restoration.

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
