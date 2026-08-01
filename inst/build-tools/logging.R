format_msg <- function(...) {
  msg <- c(...)

  if (require(cli, quietly = TRUE)) {
    msg <- cli::ansi_strwrap(msg)
    msg <- cli::boxx(msg, border_style = "double")
  } else {
    msg <- strwrap(msg)
    msg <- paste("*\t", c("", msg, ""))
    msg <- paste(msg, collapse = "\n")
    starline <- paste(rep("*", 0.9 * getOption("width")), collapse = "")
    msg <- paste(c(starline, msg, starline), collapse = "\n")
  }

  msg
}

stop2 <- function(...) {
  stop("\n", format_msg(...), call. = FALSE)
}

warning2 <- function(...) {
  warning("\n", format_msg(...), immediate. = TRUE, call. = FALSE)
}
