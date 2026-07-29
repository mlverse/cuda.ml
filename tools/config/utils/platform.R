#' utility functions that may potentially become platform-specific in future

nproc <- function() {
  # Try to run `nproc` to detect number of cores available
  tryCatch(
    as.integer(system2("nproc", stdout = TRUE, stderr = NULL)),
    error = function(e) 2L
  )
}

cuml_r_universe_build <- function() {
  identical(Sys.getenv("UNIVERSE_NAME", unset = ""), "mlverse")
}

cuml_linux_x86_64 <- function() {
  identical(Sys.info()[["sysname"]], "Linux") &&
    Sys.info()[["machine"]] %in% c("x86_64", "amd64")
}
