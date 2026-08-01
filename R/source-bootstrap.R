cuda_ml_source_build_tools <- function() {
  if (!is.null(.cuda_ml_state$build_tools)) {
    return(.cuda_ml_state$build_tools)
  }

  root <- system.file("build-tools", package = "cuda.ml")
  artifact_root <- system.file("artifacts", package = "cuda.ml")
  files <- c(
    "logging.R",
    "platform.R",
    "nvcc.R",
    "artifacts.R",
    "bootstrap.R",
    "cuml.R",
    "cmake.R"
  )
  paths <- file.path(root, files)
  if (!nzchar(root) || !nzchar(artifact_root) || any(!file.exists(paths))) {
    stop("The cuda.ml source-build tools are missing.", call. = FALSE)
  }

  tools <- new.env(parent = asNamespace("utils"))
  tools$cuml_artifact_root <- function() artifact_root
  for (path in paths) {
    sys.source(path, envir = tools, keep.source = FALSE)
  }
  tools$cuml_bootstrap_cache_dir <- function() {
    override <- Sys.getenv("CUML_BOOTSTRAP_CACHE", unset = "")
    if (nzchar(override)) {
      return(normalizePath(path.expand(override), mustWork = FALSE))
    }
    file.path(cuda_ml_cache_dir(), "source-toolchains-v1")
  }

  .cuda_ml_state$build_tools <- tools
  tools
}
