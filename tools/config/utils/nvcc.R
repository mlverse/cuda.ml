find_nvcc <- function(stop_if_missing = TRUE) {
  nvcc_candidates <- character()

  # Prefer an explicit CUDA_HOME over PATH/default discovery.
  cuda_home <- Sys.getenv("CUDA_HOME")
  if (nzchar(cuda_home)) {
    nvcc_candidates <- c(nvcc_candidates, file.path(cuda_home, "bin", "nvcc"))
  }

  nvcc_candidates <- unique(c(
    nvcc_candidates,
    "nvcc",
    "/usr/local/cuda/bin/nvcc"
  ))

  nvcc_path <- NULL
  cuda_version <- NULL
  for (candidate in nvcc_candidates) {
    version <- nvcc_version_from_path(candidate)
    if (!is.null(version)) {
      nvcc_path <- candidate
      cuda_version <- version
      break
    }
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

nvcc_supported_architectures <- function(nvcc) {
  out <- suppressWarnings(
    tryCatch(system2(nvcc, "--list-gpu-arch", stdout = TRUE, stderr = TRUE), error = function(e) NULL)
  )
  if (is.null(out)) {
    return(character())
  }

  archs <- grep("^compute_[0-9]+$", out, value = TRUE)
  unique(sub("^compute_", "", archs))
}

detected_gpu_architectures <- function() {
  out <- suppressWarnings(
    tryCatch(
      system2(
        "nvidia-smi",
        c("--query-gpu=compute_cap", "--format=csv,noheader"),
        stdout = TRUE,
        stderr = TRUE
      ),
      error = function(e) NULL
    )
  )
  if (is.null(out)) {
    return(character())
  }

  caps <- regmatches(out, gregexpr("\\b[0-9]+\\.[0-9]+\\b", out))
  caps <- unlist(caps, use.names = FALSE)
  unique(gsub("\\.", "", caps))
}

infer_cuda_architectures <- function(nvcc) {
  supported <- nvcc_supported_architectures(nvcc$path)
  detected <- detected_gpu_architectures()
  compatible <- intersect(detected, supported)

  if (length(compatible) > 0) {
    unsupported <- setdiff(detected, compatible)
    if (length(unsupported) > 0) {
      message(
        "Ignoring GPU architectures unsupported by ",
        nvcc$path,
        ": ",
        paste(unsupported, collapse = ";")
      )
    }
    return(paste(compatible, collapse = ";"))
  }

  if (length(detected) > 0 && length(supported) > 0) {
    stop2(
      paste0("Detected GPU architectures: ", paste(detected, collapse = ";")),
      paste0("Architectures supported by ", nvcc$path, ": ", paste(supported, collapse = ";")),
      "No detected GPU architecture is supported by this CUDA compiler.",
      "Install a CUDA toolkit that supports your GPU, or set CUML_CUDA_ARCHITECTURES manually."
    )
  }

  if (length(supported) > 0) {
    message(
      "Unable to detect a GPU architecture; defaulting CMAKE_CUDA_ARCHITECTURES to ",
      supported[[1]],
      ". Set CUML_CUDA_ARCHITECTURES to override."
    )
    return(supported[[1]])
  }

  "NATIVE"
}
