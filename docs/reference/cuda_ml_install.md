# Install a cuda.ml native backend

By default, downloads, verifies, extracts, and caches the precompiled
backend and its runtime libraries. Alternatively, bootstraps a locked
CUDA and RAPIDS build toolchain and compiles the native backend on the
host. Calling it again with the same inputs is a no-op. Use
`device = "cpu"` to install the separate nvForest CPU inference backend,
roughly 1 MiB to download and 3 MiB when installed, without the complete
CUDA and RAPIDS runtime. Installation never occurs implicitly during
model loading or prediction.

## Usage

``` r
cuda_ml_install(
  source = FALSE,
  dependencies = "managed",
  architectures = NULL,
  device = c("gpu", "cpu")
)
```

## Arguments

- source:

  A logical value. If `FALSE`, install the requested prebuilt backend.
  If `TRUE`, compile the complete GPU backend from the native sources
  included in the R package. Source installation is not supported for
  `device = "cpu"`.

- dependencies:

  For a source installation, either `"managed"` to download and cache
  the exact locked build dependencies, or `"host"` to use explicit host
  installations.

- architectures:

  For a source installation, `NULL`, `"native"`, `"portable"`, or an
  explicit semicolon-separated CMake CUDA architecture list. Managed
  source builds detect CUDA-visible GPUs by default and otherwise use
  the package's portable architecture list. `"native"` requires
  detection, and `"portable"` forces the package list. Host source
  builds use `CUML_CUDA_ARCHITECTURES` when this argument is `NULL`.

- device:

  Backend to install: `"gpu"` installs the complete CUDA and RAPIDS
  backend used for training and GPU inference; `"cpu"` installs only the
  CPU nvForest inference backend. CPU installation does not provision
  the complete CUDA and RAPIDS runtime or any cuML algorithms.

## Value

Invisibly returns `TRUE`.

## Details

The default cache is `tools::R_user_dir("cuda.ml", "cache")`. Set
`CUDA_ML_CACHE_DIR` to use a different cache root. Set
`CUDA_ML_BACKEND_MIRROR` to an `https://` or `file://` directory
containing the exact locked backend archive.

The CPU-only backend supports nvForest model loading, restoration, and
inference. It does not provide cuML training or GPU inference and
requires neither an NVIDIA GPU nor an NVIDIA driver. It contains no CUDA
runtime libraries. Install the complete backend separately with
`cuda_ml_install()` when training or GPU inference is needed.

A managed source installation downloads no precompiled cuda.ml backend.
It downloads and verifies the locked CUDA 13.2.2 and RAPIDS 26.06
development artifacts, CMake, and Ninja; builds Treelite 4.7.0
statically; and caches that toolchain. Only Linux x86_64 with glibc 2.28
or newer and GNU C++ 14 or newer are required on the host. When
`CUDA_ML_CXX` is unset, the installer prefers `g++-14`, then `g++`, on
`PATH`. Set `CUDA_ML_CXX` to override this discovery.

By default, a managed source build uses `nvidia-smi` to detect distinct
CUDA-visible GPU compute capabilities and compiles their real targets.
It honors `CUDA_VISIBLE_DEVICES`. If detection is unavailable, it uses
the package's portable list, so GPU-free build hosts remain supported.
Set `architectures = "native"` to require detection or
`architectures = "portable"` to force the package list. Native targets
usually reduce build time and backend size, but the resulting backend
supports only those GPU architectures.

A host source installation makes no downloads. It requires CUDA Toolkit
13.2.2 in `CUDA_HOME`; a `CUML_PREFIX` containing cuML and nvForest
26.06, Treelite 4.7.0 headers, and `lib/libtreelite_static.a`; an
explicit CMake CUDA architecture list in `CUML_CUDA_ARCHITECTURES`; and
GNU C++ 14 or newer in `CUDA_ML_CXX`. CMake 3.21.1 or newer must be on
`PATH`.

## Examples

``` r
if (FALSE) { # \dontrun{
cuda_ml_install()

cuda_ml_install(device = "cpu")

cuda_ml_install(source = TRUE)

cuda_ml_install(source = TRUE, architectures = "native")

cuda_ml_install(source = TRUE, architectures = "portable")

Sys.setenv(
  CUDA_HOME = "/usr/local/cuda-13.2",
  CUML_PREFIX = "/opt/rapids-26.06",
  CUML_CUDA_ARCHITECTURES = "86-real",
  CUDA_ML_CXX = "/usr/bin/g++-14"
)
cuda_ml_install(source = TRUE, dependencies = "host")
} # }
```
