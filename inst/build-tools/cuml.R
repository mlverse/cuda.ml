check_libcuml_path <- function(path) {
  cuml_headers_dir <- file.path(path, "include", "cuml")
  cuml_lib <- file.path(path, "lib", "libcuml.so")
  dir.exists(cuml_headers_dir) && file.exists(cuml_lib)
}

check_functional_prefix <- function(path) {
  required <- file.path(
    path,
    c(
      "include/nvforest/forest_model.hpp",
      "include/nvforest/treelite_importer.hpp",
      "include/treelite/tree.h",
      "include/treelite/version.h",
      "lib/libnvforest++.so",
      "lib/libtreelite_static.a"
    )
  )
  check_libcuml_path(path) &&
    all(file.exists(required)) &&
    identical(cuml_version_from_prefix(path), cuml_managed_rapids_version()) &&
    identical(
      nvforest_version_from_prefix(path),
      cuml_managed_nvforest_version()
    ) &&
    identical(
      treelite_version_from_prefix(path),
      cuml_managed_treelite_version()
    )
}

cuml_version_from_prefix <- function(path) {
  version_header <- file.path(path, "include", "cuml", "version_config.hpp")
  if (!file.exists(version_header)) {
    return(NA_character_)
  }

  lines <- readLines(version_header, warn = FALSE)
  read_component <- function(component) {
    pattern <- paste0(
      "^#define[[:space:]]+CUML_VERSION_",
      component,
      "[[:space:]]+"
    )
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
  stopifnot(
    is.list(nvcc),
    length(nvcc$version) == 1L,
    is.character(cuml_prefix)
  )

  cuda_version <- paste(nvcc$version$major, nvcc$version$minor, sep = ".")
  if (!identical(cuda_version, cuml_managed_cuda_version())) {
    stop2(
      paste0("CUDA ", cuda_version, " is not supported by this build."),
      paste0(
        "Use the pinned CUDA ",
        cuml_managed_cuda_version(),
        " toolchain."
      )
    )
  }

  component_version <- nvcc_component_version_from_path(nvcc$path)
  if (!identical(component_version, cuml_managed_cuda_component_version())) {
    stop2(
      paste0(
        "CUDA compiler component ",
        component_version,
        " is not supported."
      ),
      paste0("Use nvcc ", cuml_managed_cuda_component_version(), ".")
    )
  }

  rapids_version <- cuml_version_from_prefix(cuml_prefix)
  if (!identical(rapids_version, cuml_managed_rapids_version())) {
    stop2(
      paste0(
        "RAPIDS cuML ",
        rapids_version,
        " is not supported by this build."
      ),
      paste0(
        "Use the pinned RAPIDS cuML ",
        cuml_managed_rapids_version(),
        " headers and libraries."
      )
    )
  }

  invisible(TRUE)
}
