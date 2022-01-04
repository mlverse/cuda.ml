find_nvcc <- function(stop_if_missing = TRUE) {
  # Check if nvcc from path is available
  nvcc_path <- "nvcc"
  cuda_version <- nvcc_version_from_path(nvcc_path)

  # Check if nvcc from CUDA_HOME is available
  cuda_home <- Sys.getenv("CUDA_HOME")
  if (nzchar(cuda_home) && is.null(cuda_version)) {
    nvcc_path <- file.path(cuda_home, "bin", "nvcc")
    cuda_version <- nvcc_version_from_path(nvcc_path)
  }

  # Check nvcc from default install location.
  if (is.null(cuda_version)) {
    nvcc_path <- "/usr/local/cuda/bin/nvcc"
    cuda_version <- nvcc_version_from_path(nvcc_path)
  }

  # No nvcc found! Error!
  if (is.null(cuda_version)) {
    if (stop_if_missing) {
      stop2(
        "Unable to locate a CUDA compiler (nvcc).",
        "Please ensure it is present in PATH (e.g., run",
        "`export PATH=\"${PATH}:/usr/local/cuda/bin\"` or",
        "similar) and try again."
      )
    } else {
      return(NULL)
    }
  }

  # Nvcc found but wrong cuda version.
  minimum_supported <- package_version("11.0")
  if (cuda_version < minimum_supported) {
    stop2(
      paste0("Found nvcc '", nvcc_path, "'"),
      paste0("CUDA version '", cuda_version, "' is not supported."),
      paste0("The minimum required version is '", minimum_supported, "'")
    )
  }

  # return nvcc path.
  return(list(path = nvcc_path, version = cuda_version))
}

nvcc_version_from_path <- function(nvcc) {
  suppressWarnings(
    nvcc <- tryCatch(system2(nvcc, "--version", stdout = TRUE, stderr = TRUE), error = function(e) NULL)
  )

  if (is.null(nvcc) || !any(grepl("release", nvcc))) {
    return(NULL)
  }

  version <- gsub(".*release |, V.*", "", nvcc[grepl("release", nvcc)])
  package_version(version)
}
