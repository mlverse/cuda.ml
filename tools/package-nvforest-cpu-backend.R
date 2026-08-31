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
  "cuda.ml-nvforest-cpu-",
  description[["Version"]],
  "-r",
  r_version,
  "-",
  platform,
  ".tar.gz"
)
archive <- file.path(output_dir, filename)
staging <- tempfile("cuda-ml-nvforest-cpu-", tmpdir = output_dir)
dir.create(staging)
on.exit(unlink(staging, recursive = TRUE, force = TRUE), add = TRUE)

system_library <- function(library) {
  grepl(
    paste0(
      "^(lib(R|c|m|dl|pthread|rt|gcc_s|stdc\\+\\+|z|bz2|lzma|crypt|util|",
      "resolv)\\.so(\\..*)?|ld-linux-x86-64\\.so\\.2)$"
    ),
    library
  )
}

needed <- function(path) {
  dynamic <- system2(
    "readelf",
    c("-d", shQuote(path)),
    stdout = TRUE,
    stderr = TRUE
  )
  stopifnot(is.null(attr(dynamic, "status")))
  lines <- grep("\\(NEEDED\\)", dynamic, value = TRUE)
  unname(sub(".*Shared library: \\[([^]]+)\\].*", "\\1", lines))
}

dependencies <- needed(backend)
non_system_dependencies <- dependencies[
  !vapply(dependencies, system_library, logical(1))
]
stopifnot(length(non_system_dependencies) == 0L)

backend_name <- paste0("cuda.ml.nvforest", .Platform$dynlib.ext)
stopifnot(
  identical(.Platform$dynlib.ext, ".so"),
  file.copy(backend, file.path(staging, backend_name), copy.mode = TRUE)
)

license_sources <- c(
  "cuda.ml" = "LICENSE.md",
  "cuda.ml dependency provenance" = file.path("inst", "COPYRIGHTS"),
  "Treelite 4.7.0" = file.path(
    "inst",
    "third-party",
    "treelite-LICENSE"
  ),
  "RapidJSON ab1842a2dae061284c0a62dca1cc6d5e7e37e346" = file.path(
    "inst",
    "third-party",
    "rapidjson-LICENSE"
  ),
  "nlohmann/json 3.11.3" = file.path(
    "inst",
    "third-party",
    "nlohmann-json-LICENSE"
  ),
  "mdspan 0.6.0" = file.path(
    "inst",
    "third-party",
    "mdspan-LICENSE"
  ),
  "nvForest 26.06.0" = file.path(
    "inst",
    "third-party",
    "nvforest-LICENSE"
  )
)
stopifnot(
  !anyDuplicated(names(license_sources)),
  all(file.exists(license_sources)),
  all(!file.info(license_sources)[["isdir"]])
)
separator <- paste(rep("=", 78L), collapse = "")
license_sections <- lapply(seq_along(license_sources), function(index) {
  c(
    separator,
    names(license_sources)[[index]],
    separator,
    "",
    readLines(license_sources[[index]], warn = FALSE),
    ""
  )
})
license_name <- "THIRD-PARTY-LICENSES.txt"
license_path <- file.path(staging, license_name)
writeLines(unlist(license_sections, use.names = FALSE), license_path)
stopifnot(file.exists(license_path), file.info(license_path)[["size"]] > 0)

write.dcf(
  matrix(
    c(
      "1",
      "cuda.ml",
      "nvforest-cpu",
      description[["Version"]],
      r_version,
      platform,
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
        "Backend",
        "Package-Version",
        "R-Version",
        "Platform",
        "nvForest",
        "Treelite",
        "Backend-SHA256",
        "Source-Commit"
      )
    )
  ),
  file = file.path(staging, "backend.dcf")
)

files <- sort(c("backend.dcf", backend_name, license_name))
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
    shQuote(files)
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
  package_version = description[["Version"]],
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
