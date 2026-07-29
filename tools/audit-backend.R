args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L, file.exists(args[[1L]]))

cuobjdump <- normalizePath(args[[1L]], mustWork = TRUE)
backend <- system.file(
  "libs",
  paste0("cuda.ml", .Platform$dynlib.ext),
  package = "cuda.ml"
)
stopifnot(nzchar(backend), file.exists(backend))

elf <- system2(
  cuobjdump,
  c("--list-elf", shQuote(backend)),
  stdout = TRUE,
  stderr = TRUE
)
ptx <- system2(
  cuobjdump,
  c("--list-ptx", shQuote(backend)),
  stdout = TRUE,
  stderr = TRUE
)
stopifnot(is.null(attr(elf, "status")), is.null(attr(ptx, "status")))

targets <- function(output, suffix) {
  lines <- grep(paste0("[.]sm_[0-9]+[.]", suffix, "$"), output, value = TRUE)
  unique(sub(
    paste0(".*[.](sm_[0-9]+)[.]", suffix, "$"),
    "\\1",
    lines
  ))
}

sass <- targets(elf, "cubin")
virtual <- targets(ptx, "ptx")
stopifnot(
  setequal(
    sass,
    c("sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120")
  ),
  identical(virtual, "sm_120")
)

dynamic <- system2(
  "readelf",
  c("-d", shQuote(backend)),
  stdout = TRUE,
  stderr = TRUE
)
stopifnot(is.null(attr(dynamic, "status")))
rpath <- grep("\\((RPATH|RUNPATH)\\)", dynamic, value = TRUE)
stopifnot(length(rpath) == 1L, grepl("\\[\\$ORIGIN\\]", rpath))

needed <- grep("\\(NEEDED\\)", dynamic, value = TRUE)
needed <- sub(".*Shared library: \\[([^]]+)\\].*", "\\1", needed)
system <- grepl(
  "^(lib(R|c|m|dl|pthread|rt|gcc_s|stdc\\+\\+)\\.so(\\..*)?|ld-linux-x86-64\\.so\\.2)$",
  needed
)
stopifnot(
  setequal(
    needed[!system],
    c("libcuml.so", "libcudart.so.13")
  )
)

message("cuda.ml backend audit passed")
