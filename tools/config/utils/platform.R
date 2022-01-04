#' utility functions that may potentially become platform-specific in future

nproc <- function() {
  # Try to run `nproc` to detect number of cores available
  tryCatch(
    as.integer(system2("nproc", stdout = TRUE, stderr = NULL)),
    error = function(e) 2L
  )
}
