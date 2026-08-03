args <- commandArgs(trailingOnly = TRUE)
stopifnot(
  length(args) == 4L,
  file.exists(args[[1L]]),
  dir.exists(args[[2L]]),
  grepl("^[[:xdigit:]]{40}$", args[[3L]]),
  startsWith(args[[4L]], "https://"),
  !endsWith(args[[4L]], "/")
)

backend <- normalizePath(args[[1L]], mustWork = TRUE)
output_dir <- normalizePath(args[[2L]], mustWork = TRUE)
source_commit <- args[[3L]]
base_url <- args[[4L]]
description <- read.dcf("DESCRIPTION")[1L, , drop = TRUE]
metadata <- read.dcf("inst/cuda-ml-backend.dcf")[1L, , drop = TRUE]
r_version <- paste(
  R.version$major,
  strsplit(R.version$minor, ".", fixed = TRUE)[[1L]][[1L]],
  sep = "."
)
backend_hash <- unname(digest::digest(
  file = backend,
  algo = "sha256",
  serialize = FALSE
))
platform <- unname(metadata[["Platform"]])
filename <- paste0(
  "cuda.ml-",
  description[["Version"]],
  "-r",
  r_version,
  "-",
  platform,
  "-cu13.tar.gz"
)
archive <- file.path(output_dir, filename)
staging <- tempfile("cuda-ml-backend-", tmpdir = output_dir)
dir.create(staging)
on.exit(unlink(staging, recursive = TRUE, force = TRUE), add = TRUE)

backend_name <- paste0("cuda.ml", .Platform$dynlib.ext)
stopifnot(
  identical(.Platform$dynlib.ext, ".so"),
  file.copy(backend, file.path(staging, backend_name), copy.mode = TRUE)
)
write.dcf(
  matrix(
    c(
      "1",
      "cuda.ml",
      description[["Version"]],
      r_version,
      platform,
      metadata[["CUDA"]],
      metadata[["RAPIDS"]],
      metadata[["nvForest"]],
      metadata[["Treelite"]],
      backend_hash,
      source_commit
    ),
    nrow = 1L,
    dimnames = list(
      NULL,
      c(
        "Schema",
        "Package",
        "Package-Version",
        "R-Version",
        "Platform",
        "CUDA",
        "RAPIDS",
        "nvForest",
        "Treelite",
        "Backend-SHA256",
        "Source-Commit"
      )
    )
  ),
  file = file.path(staging, "backend.dcf")
)

old <- setwd(staging)
on.exit(setwd(old), add = TRUE)
status <- system2(
  "tar",
  c(
    "--sort=name",
    "--mtime=@0",
    "--owner=0",
    "--group=0",
    "--numeric-owner",
    "-czf",
    shQuote(archive),
    "backend.dcf",
    backend_name
  )
)
stopifnot(identical(status, 0L), file.exists(archive))

archive_hash <- unname(digest::digest(
  file = archive,
  algo = "sha256",
  serialize = FALSE
))
row <- data.frame(
  r_version = r_version,
  filename = filename,
  url = paste0(base_url, "/", filename),
  size = as.numeric(file.info(archive)[["size"]]),
  sha256 = archive_hash,
  backend_sha256 = backend_hash,
  stringsAsFactors = FALSE
)
row_path <- file.path(output_dir, paste0(filename, ".row.tsv"))
utils::write.table(
  row,
  file = row_path,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
message(archive)
message(row_path)
