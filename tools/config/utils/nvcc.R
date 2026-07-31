nvcc_version_from_path <- function(nvcc) {
  suppressWarnings(
    nvcc <- tryCatch(
      system2(nvcc, "--version", stdout = TRUE, stderr = TRUE),
      error = function(e) NULL
    )
  )

  if (is.null(nvcc) || !any(grepl("release", nvcc))) {
    return(NULL)
  }

  version <- gsub(".*release |, V.*", "", nvcc[grepl("release", nvcc)])
  package_version(version)
}
