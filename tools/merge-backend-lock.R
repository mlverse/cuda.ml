args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L, dir.exists(args[[2L]]))

destination <- args[[1L]]
rows <- list.files(
  args[[2L]],
  pattern = "[.]row[.]tsv$",
  recursive = TRUE,
  full.names = TRUE
)
stopifnot(length(rows) > 0L)
catalog <- do.call(
  rbind,
  lapply(
    rows,
    utils::read.delim,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    colClasses = c(r_version = "character")
  )
)
required <- c(
  "r_version",
  "filename",
  "url",
  "size",
  "sha256",
  "backend_sha256"
)
expected_r_versions <- paste0("4.", 1:6)
stopifnot(
  identical(names(catalog), required),
  setequal(catalog$r_version, expected_r_versions),
  !anyDuplicated(catalog$r_version),
  all(file.exists(file.path(args[[2L]], catalog$filename)))
)
catalog <- catalog[order(package_version(catalog$r_version)), , drop = FALSE]
dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
utils::write.table(
  catalog,
  file = destination,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
