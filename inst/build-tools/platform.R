cuml_build_mode <- function() {
  mode <- Sys.getenv("CUDA_ML_BUILD_MODE", unset = "")
  if (nzchar(mode)) {
    if (!mode %in% c("managed", "stub", "local")) {
      stop2("CUDA_ML_BUILD_MODE must be one of: managed, stub, local.")
    }
    return(mode)
  }
  "stub"
}

cuml_linux_x86_64 <- function() {
  identical(Sys.info()[["sysname"]], "Linux") &&
    Sys.info()[["machine"]] %in% c("x86_64", "amd64")
}

cuml_glibc_version <- function() {
  output <- suppressWarnings(tryCatch(
    system2("getconf", "GNU_LIBC_VERSION", stdout = TRUE, stderr = FALSE),
    error = function(e) character()
  ))
  match <- regexec("^glibc ([0-9]+[.][0-9]+)$", output)
  values <- regmatches(output, match)
  if (length(values) != 1L || length(values[[1L]]) != 2L) {
    return(NA_character_)
  }
  values[[1L]][[2L]]
}

cuml_manylinux_2_28_x86_64 <- function() {
  cuml_linux_x86_64() && identical(cuml_glibc_version(), "2.28")
}

cuml_supported_local_platform <- function() {
  glibc <- cuml_glibc_version()
  cuml_linux_x86_64() &&
    !is.na(glibc) &&
    package_version(glibc) >= package_version("2.28")
}
