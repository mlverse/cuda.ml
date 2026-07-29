check_libcuml_path <- function(path) {
  cuml_headers_dir <- file.path(path, "include", "cuml")
  cuml_libs <- file.path(path, "lib", c("libcuml.so", "libcuml++.so"))
  dir.exists(cuml_headers_dir) && any(file.exists(cuml_libs))
}

cuml_version_from_prefix <- function(path) {
  version_header <- file.path(path, "include", "cuml", "version_config.hpp")
  if (!file.exists(version_header)) {
    return(NA_character_)
  }

  lines <- readLines(version_header, warn = FALSE)
  read_component <- function(component) {
    pattern <- paste0("^#define[[:space:]]+CUML_VERSION_", component, "[[:space:]]+")
    line <- grep(pattern, lines, value = TRUE)
    if (length(line) != 1L) {
      return(NA_integer_)
    }
    as.integer(sub(pattern, "", line))
  }

  major <- read_component("MAJOR")
  minor <- read_component("MINOR")
  if (is.na(major) || is.na(minor)) {
    return(NA_character_)
  }

  sprintf("%d.%02d", major, minor)
}

validate_managed_build_versions <- function(nvcc, cuml_prefix) {
  stopifnot(is.list(nvcc), length(nvcc$version) == 1L, is.character(cuml_prefix))

  cuda_version <- paste(nvcc$version$major, nvcc$version$minor, sep = ".")
  if (!identical(cuda_version, cuml_managed_cuda_version())) {
    stop2(
      paste0("CUDA ", cuda_version, " is not supported by this build."),
      paste0(
        "Use the pinned CUDA ", cuml_managed_cuda_version(),
        " toolchain."
      )
    )
  }

  rapids_version <- cuml_version_from_prefix(cuml_prefix)
  if (!identical(rapids_version, cuml_managed_rapids_version())) {
    stop2(
      paste0("RAPIDS cuML ", rapids_version, " is not supported by this build."),
      paste0(
        "Use the pinned RAPIDS cuML ", cuml_managed_rapids_version(),
        " headers and libraries."
      )
    )
  }

  invisible(TRUE)
}

get_cuml_prefix <- function() {
  cuml_prefix <- Sys.getenv("CUML_PREFIX", unset = NA_character_)
  if (is.na(cuml_prefix) || !nzchar(cuml_prefix)) {
    return(NA_character_)
  }

  normalizePath(cuml_prefix, mustWork = FALSE)
}
