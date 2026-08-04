# cuda.ml

This package provides a R interface for the RAPIDS cuML library.

## Installation

Install the portable R package from CRAN, then explicitly download its
locked native backend and runtime:

    install.packages("cuda.ml")
    cuda.ml::cuda_ml_install()

The CRAN package contains no compiled code. Prebuilt backends support
Linux x86_64 with glibc 2.28 or newer and are selected for the current R
minor version.

Loading cuda.ml does not require a GPU, load native code, create a
cache, or contact the network. Call
[`cuda_ml_install()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_install.md)
to install the complete backend used for GPU training and inference. It
uses the pinned CUDA 13.2.2 and RAPIDS cuML and nvForest 26.06 runtime
and is roughly 1.6 GiB. GPU operations then require a supported NVIDIA
GPU and driver 580 or newer.

For CPU-only nvForest inference, call `cuda_ml_install(device = "cpu")`
instead. This installs a separate backend that is roughly 1 MiB to
download and 3 MiB when installed. It does not install cuML or the
complete managed CUDA and RAPIDS runtime and requires neither an NVIDIA
GPU nor an NVIDIA driver. It contains no CUDA runtime libraries.
Treelite 4.7.0 is linked statically into both backends.

Random forests trained on a GPU by
[`cuda_ml_rand_forest()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_rand_forest.md)
can be persisted with
[`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
and restored for CPU inference with
`cuda_ml_unserialize(state, device = "cpu")`. Current nvForest model
states do not encode their deployment device. Alternatively,
[`cuda_ml_nvforest_export()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md)
writes a standard Treelite checkpoint and cuda.ml JSON metadata that can
be restored with
[`cuda_ml_nvforest_import()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md).
Native operations fail with an installation instruction until their
corresponding backend is installed. Set `CUDA_ML_CACHE_DIR` to override
the default cache and `CUDA_ML_BACKEND_MIRROR` to use an exact backend
mirror.

To compile cuda.ml itself on the host without a prebuilt cuda.ml
backend, call `cuda_ml_install(source = TRUE)`. The default managed
source build downloads the locked CUDA, RAPIDS, CMake, Ninja, and
Treelite build inputs. It detects CUDA-visible GPU architectures when
available and otherwise uses the package's portable architecture list.
Only Linux x86_64 with glibc 2.28 or newer and GNU C++ 14 or newer are
required on the host. Use `dependencies = "host"` with explicit
`CUDA_HOME`, `CUML_PREFIX`, `CUML_CUDA_ARCHITECTURES`, and `CUDA_ML_CXX`
inputs for a fully native, network-free source build.

## See also

Useful links:

- <https://mlverse.github.io/cuda.ml/>

- Report bugs at <https://github.com/mlverse/cuda.ml/issues>

## Author

Yitao Li \<yitao@rstudio.com\>
