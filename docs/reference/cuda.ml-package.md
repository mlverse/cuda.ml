# cuda.ml

This package provides a R interface for the RAPIDS cuML library.

## Installation

Install the functional Ubuntu 26.04 (Resolute) x86_64 binary from the
mlverse [R-universe Linux binary
repository](https://docs.r-universe.dev/install/binaries.html):

    linux_binary_repo <- function(universe) {
      r_version <- paste(
        R.version$major,
        strsplit(R.version$minor, ".", fixed = TRUE)[[1L]][1L],
        sep = "."
      )
      paste0(
        "https://", universe,
        ".r-universe.dev/bin/linux/resolute-",
        R.version$arch,
        "/",
        r_version,
        "/"
      )
    }

    repos <- c(
      mlverse = linux_binary_repo("mlverse"),
      CRAN = linux_binary_repo("cran")
    )
    stopifnot(
      identical(unname(Sys.info()[["sysname"]]), "Linux"),
      identical(R.version$arch, "x86_64"),
      grepl(
        "/bin/linux/resolute-x86_64/[0-9]+[.][0-9]+/$",
        repos[["mlverse"]]
      )
    )

    install.packages(
      "cuda.ml",
      repos = repos
    )

The binary repository path selects the prebuilt tarball. Stock Linux R
does not support `type = "binary"`, so leave `type` at its default. This
binary also supports WSL2 running Ubuntu 26.04. Other Linux
distributions are not yet supported by the managed binary.

Loading cuda.ml does not require a GPU, load native code, create a
cache, or contact the network. Call
[`cuda_ml_install()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_install.md)
to download and cache the pinned CUDA 13.2.2, RAPIDS cuML and nvForest
26.06, and Treelite 4.6.1 runtime while preparing a container or machine
image. Native operations fail with an installation instruction until
that explicit setup step has completed. GPU operations then require a
supported NVIDIA GPU and driver 580 or newer; nvForest CPU inference
does not.

CRAN builds are network-free source stubs. A stub reports
`backend = "stub"` from
[`cuda_ml_backend_info()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_backend_info.md);
install the R-universe binary for a no-compiler setup. Local functional
source builds require exact CUDA 13.2.2, GNU C++ 14 or newer, cuML and
nvForest 26.06, and Treelite 4.6.1 inputs. Set `CUDA_ML_CACHE_DIR` to
override the default managed runtime cache.

## See also

Useful links:

- <https://mlverse.github.io/cuda.ml/>

- Report bugs at <https://github.com/mlverse/cuda.ml/issues>

## Author

Yitao Li \<yitao@rstudio.com\>
