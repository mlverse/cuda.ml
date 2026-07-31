cuml_build_mode <- function() {
  mode <- Sys.getenv("CUDA_ML_BUILD_MODE", unset = "")
  if (nzchar(mode)) {
    if (!mode %in% c("managed", "stub", "local")) {
      stop2("CUDA_ML_BUILD_MODE must be one of: managed, stub, local.")
    }
    return(mode)
  }
  if (identical(Sys.getenv("UNIVERSE_NAME", unset = ""), "mlverse")) {
    if (cuml_ubuntu_2604_x86_64()) "managed" else "stub"
  } else {
    "stub"
  }
}

cuml_linux_x86_64 <- function() {
  identical(Sys.info()[["sysname"]], "Linux") &&
    Sys.info()[["machine"]] %in% c("x86_64", "amd64")
}

cuml_os_release_value <- function(name) {
  path <- "/etc/os-release"
  if (!file.exists(path)) {
    return("")
  }
  lines <- readLines(path, warn = FALSE)
  values <- lines[startsWith(lines, paste0(name, "="))]
  if (length(values) != 1L) {
    return("")
  }
  value <- substring(values, nchar(name) + 2L)
  sub('^"(.*)"$', "\\1", value)
}

cuml_ubuntu_2604_x86_64 <- function() {
  cuml_linux_x86_64() &&
    identical(cuml_os_release_value("ID"), "ubuntu") &&
    identical(cuml_os_release_value("VERSION_ID"), "26.04")
}
