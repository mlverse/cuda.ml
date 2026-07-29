#' Options:
#'
#' CUML_PREFIX: Required for a functional local source build. Set this to the
#'              explicit prefix containing the compatible RAPIDS cuML headers
#'              and libraries. Local builds never provision a toolchain.
#'
#' CUML_BOOTSTRAP_CACHE: Override the temporary build-toolchain cache used by
#'                       the managed mlverse R-universe build.
#'
#' CUML_CUDA_ARCHITECTURES: Override CMAKE_CUDA_ARCHITECTURES. Setting this
#'                          enables cross-compilation without a visible GPU.
#'                          Otherwise, defaults to detected GPU architectures
#'                          supported by nvcc.
#'
#' DISABLE_PARALLEL_BUILD: Parallel build using max($(nproc) - 1, 1) cores is
#'                         enabled by default but can be disabled by setting
#'                         this env variable.
#'
#' CUML4R_CMAKE_PARALLEL_LEVEL: If not set and parallel build is enabled, then
#'                              max($(nproc) - 1, 1) cores will be used by the
#'                              build process. If set, then the number of cores
#'                              specified will be used.
#'
#' UNIVERSE_NAME: When set to "mlverse", configure provisions the pinned CUDA
#'                13.2 / RAPIDS 26.06 build toolchain and builds the portable
#'                Linux x86_64 backend without requiring a visible GPU.

pkg_root <- function() {
  # devtools::load_all() might run the config script from the `src` directory.
  for (p in list(".", "..")) {
    if (file.exists(file.path(p, "DESCRIPTION"))) {
      return(normalizePath(p))
    }
  }

  # should never reach here
  pkg_root <- normalizePath(".")
  warning(
    "Unable to locate 'DESCRIPTION' file! Assuming pkg root is '",
    pkg_root, "'."
  )
  return(pkg_root)
}

load_util_fns <- function() {
  wd <- file.path(pkg_root(), "tools", "config", "utils")

  for (f in c("logging.R", "platform.R", "nvcc.R", "bootstrap.R", "cuml.R", "cmake.R")) {
    source(file.path(wd, f))
  }
}

load_util_fns()

clear_build_artifacts <- function() {
  paths <- c(
    "Makevars",
    "Makefile",
    "_deps",
    ".cmake-build",
    "CMakeCache.txt",
    "CMakeFiles",
    "cmake_install.cmake",
    "CMakeLists.txt",
    "symbols.rds",
    "*.o",
    "*.so"
  )
  for (path in paths) {
    unlink(file.path(pkg_root(), "src", path), recursive = TRUE, expand = TRUE)
  }
}

clear_build_artifacts()

write_backend_metadata <- function(
  backend,
  cuda = "",
  rapids = "",
  architectures = ""
) {
  stopifnot(backend %in% c("full", "stub"))

  path <- file.path(pkg_root(), "inst", "cuda-ml-backend.dcf")
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(
    c(
      "Schema: 1",
      paste0("Backend: ", backend),
      paste0("CUDA: ", cuda),
      paste0("RAPIDS: ", rapids),
      paste0("Architectures: ", architectures)
    ),
    path
  )
}

run_cmake <- function(nvcc, cuml_prefix, cuda_architectures) {
  stopifnot(
    is.list(nvcc),
    is.character(cuml_prefix),
    length(cuml_prefix) == 1L,
    is.character(cuda_architectures),
    length(cuda_architectures) == 1L
  )

  wd <- getwd()
  on.exit(setwd(wd))
  setwd(pkg_root())

  define(R_INCLUDE_DIR = R.home("include"))
  define(RCPP_INCLUDE_DIR = system.file("include", package = "Rcpp"))
  configure_file(file.path("src", "CMakeLists.txt.in"))

  cmake_bin <- find_or_download_cmake(
    min_version = cuda_ml_min_cmake_version,
    exdir = file.path(pkg_root(), "tools")
  )
  src_dir <- normalizePath(file.path(pkg_root(), "src"))
  build_dir <- file.path(src_dir, ".cmake-build")
  dir.create(build_dir, recursive = TRUE, showWarnings = FALSE)

  define(
    CMAKE_BIN = shQuote(cmake_bin),
    CMAKE_BUILD_DIR = shQuote(build_dir),
    CMAKE_BUILD_OUTPUT = shQuote(file.path(build_dir, "cuda.ml.so")),
    RSCRIPT_BIN = shQuote(file.path(R.home("bin"), "Rscript"))
  )
  configure_file(
    file.path("tools", "config", "Makefile.cmake.in"),
    target = file.path("src", "Makefile")
  )

  stopifnot(!is.na(cuml_prefix), nzchar(cuml_prefix))
  cmake_prefix_path <- c(
    Sys.getenv("CMAKE_PREFIX_PATH", unset = ""),
    cuml_prefix
  )
  cmake_prefix_path <- paste(cmake_prefix_path[nzchar(cmake_prefix_path)], collapse = ":")
  Sys.setenv(CMAKE_PREFIX_PATH = cmake_prefix_path)

  cmake_args <- c(
    "-S", src_dir,
    "-B", build_dir,
    paste0("-DCMAKE_CUDA_ARCHITECTURES=", cuda_architectures),
    paste0("-DCUML_INCLUDE_DIR=", file.path(cuml_prefix, "include")),
    paste0("-DCUML_LIB_DIR=", file.path(cuml_prefix, "lib")),
    paste0("-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=", build_dir),
    paste0(
      "-DCUML_STUB_HEADERS_DIR=", normalizePath(file.path(src_dir, "stubs"))
    ),
    paste0("-DCMAKE_CUDA_COMPILER=", nvcc$path),
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=TRUE"
  )
  cmake_args <- c(
    cmake_args,
    "-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=TRUE",
    "-DCMAKE_INSTALL_RPATH:STRING=$ORIGIN",
    "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=FALSE"
  )
  rc <- system2(cmake_bin, args = shQuote(cmake_args))

  if (rc != 0) {
    stop("Failed to run 'cmake'!")
  }
}

nvcc <- NULL
cuml_prefix <- NA_character_
cuda_architectures <- NA_character_

if (cuml_r_universe_build() && cuml_linux_x86_64()) {
  managed_build <- bootstrap_managed_build_from_pip()
  nvcc <- managed_build$nvcc
  cuml_prefix <- managed_build$prefix
  cuda_architectures <- cuml_managed_cuda_architectures()
} else if (cuml_cran_like() || cuml_r_universe_build()) {
  # CRAN-like and unsupported R-universe builds are network-free stubs.
} else {
  nvcc <- find_nvcc(stop_if_missing = FALSE)
  if (is.null(nvcc)) {
    if (nzchar(Sys.getenv("CUML_PREFIX", unset = ""))) {
      stop2(
        "`CUML_PREFIX` was supplied, but a CUDA compiler was not found.",
        "Supply the pinned CUDA 13.2 toolkit through `CUDA_HOME`."
      )
    } else {
      warn_missing_nvcc()
    }
  } else {
    cuml_prefix <- get_cuml_prefix()
    if (is.na(cuml_prefix)) {
      warning2(
        "A functional local source build requires an explicit `CUML_PREFIX`.",
        "Set it to a RAPIDS cuML 26.06 prefix containing `include/cuml`",
        "and `lib/libcuml.so`. Falling back to a stub-only build."
      )
    } else if (!check_libcuml_path(cuml_prefix)) {
      stop2(
        paste0("Invalid CUML_PREFIX: ", cuml_prefix),
        "Expected `include/cuml` and `lib/libcuml.so`."
      )
    } else {
      cuda_architectures <- Sys.getenv("CUML_CUDA_ARCHITECTURES", unset = NA)
      if (is.na(cuda_architectures)) {
        cuda_architectures <- infer_cuda_architectures(nvcc)
      }
    }
  }
}

full_build <- !is.null(nvcc) &&
  !is.na(cuml_prefix) &&
  check_libcuml_path(cuml_prefix)

if (!full_build) {
  wd <- getwd()
  on.exit(setwd(wd))
  setwd(pkg_root())
  define(STUBS_HEADERS_DIR = normalizePath(file.path(getwd(), "src", "stubs")))
  write_backend_metadata("stub")
} else {
  validate_managed_build_versions(nvcc, cuml_prefix)
  define(STUBS_HEADERS_DIR = "")
  n_jobs <- (
    if (!is.na(Sys.getenv("DISABLE_PARALLEL_BUILD", unset = NA))) {
      1L
    } else {
      user_specified_parallel_level <- Sys.getenv("CUML4R_CMAKE_PARALLEL_LEVEL", unset = NA)
      if (!is.na(user_specified_parallel_level)) {
        as.integer(user_specified_parallel_level)
      } else {
        max(nproc() - 1L, 1L)
      }
    })
  define(CMAKE_BUILD_PARALLEL_ARGS = paste("--parallel", n_jobs))

  run_cmake(nvcc, cuml_prefix, cuda_architectures)
  write_backend_metadata(
    backend = "full",
    cuda = cuml_managed_cuda_version(),
    rapids = cuml_managed_rapids_version(),
    architectures = cuda_architectures
  )
}
