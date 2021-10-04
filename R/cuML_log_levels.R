match_cuML_log_level <- function(log_level = c("off", "critical", "error", "warn", "info", "debug", "trace")) {
  log_level <- match.arg(log_level)

  switch(log_level,
    off = 0L,
    critical = 1L,
    error = 2L,
    warn = 3L,
    info = 4L,
    debug = 5L,
    trace = 6L
  )
}
