args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L, file.exists(args[[1L]]))

backend <- normalizePath(args[[1L]], mustWork = TRUE)
libdir <- dirname(backend)
license_name <- "THIRD-PARTY-LICENSES.txt"
license_path <- file.path(libdir, license_name)
license_info <- file.info(license_path)
stopifnot(identical(
  basename(backend),
  paste0(
    "cuda.ml.nvforest",
    .Platform$dynlib.ext
  )
))
stopifnot(
  file.exists(license_path),
  !is.na(license_info[["isdir"]]),
  !license_info[["isdir"]],
  license_info[["size"]] > 0,
  !nzchar(Sys.readlink(license_path))
)
license_lines <- readLines(license_path, warn = FALSE)
required_license_sections <- c(
  "cuda.ml",
  "cuda.ml dependency provenance",
  "Treelite 4.7.0",
  "RapidJSON ab1842a2dae061284c0a62dca1cc6d5e7e37e346",
  "nlohmann/json 3.11.3",
  "mdspan 0.6.0",
  "nvForest 26.06.0"
)
stopifnot(all(required_license_sections %in% license_lines))

system_library <- function(library) {
  grepl(
    paste0(
      "^(lib(R|c|m|dl|pthread|rt|gcc_s|stdc\\+\\+|z|bz2|lzma|crypt|util|",
      "resolv)\\.so(\\..*)?|ld-linux-x86-64\\.so\\.2)$"
    ),
    library
  )
}

dynamic_section <- function(path) {
  output <- system2(
    "readelf",
    c("-d", shQuote(path)),
    stdout = TRUE,
    stderr = TRUE
  )
  stopifnot(is.null(attr(output, "status")))
  output
}

needed <- function(path) {
  lines <- grep("\\(NEEDED\\)", dynamic_section(path), value = TRUE)
  unname(sub(".*Shared library: \\[([^]]+)\\].*", "\\1", lines))
}

dependencies <- needed(backend)
non_system_dependencies <- dependencies[
  !vapply(dependencies, system_library, logical(1))
]
stopifnot(length(non_system_dependencies) == 0L)

available <- list.files(
  libdir,
  full.names = TRUE,
  all.files = TRUE,
  no.. = TRUE
)
available <- available[
  !basename(available) %in% c("backend.dcf", license_name)
]
stopifnot(
  length(available) == 1L,
  identical(normalizePath(available, mustWork = TRUE), backend)
)

dynamic <- dynamic_section(backend)
rpath <- grep("\\((RPATH|RUNPATH)\\)", dynamic, value = TRUE)
stopifnot(length(rpath) <= 1L, !length(rpath) || grepl("\\$ORIGIN", rpath))

library(Rcpp)
stopifnot(!"cuda.ml.nvforest" %in% names(getLoadedDLLs()))
dll <- dyn.load(backend, local = TRUE, now = TRUE)
on.exit(dyn.unload(dll[["path"]]), add = TRUE)

registered <- getDLLRegisteredRoutines(dll)[[".Call"]]
registered_manifest <- data.frame(
  symbol = names(registered),
  arity = as.integer(vapply(registered, `[[`, numeric(1), "numParameters")),
  stringsAsFactors = FALSE
)
registered_manifest <- registered_manifest[
  order(registered_manifest$symbol),
  ,
  drop = FALSE
]
rownames(registered_manifest) <- NULL
expected_manifest <- utils::read.delim(
  file.path("inst", "nvforest-native-symbols.txt"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
expected_manifest <- expected_manifest[
  order(expected_manifest$symbol),
  ,
  drop = FALSE
]
rownames(expected_manifest) <- NULL
stopifnot(identical(registered_manifest, expected_manifest))

version_symbol <- getNativeSymbolInfo(
  "_cuda_ml_nvforest_backend_versions",
  PACKAGE = dll,
  withRegistrationInfo = TRUE
)
versions <- do.call(.Call, list(version_symbol))
metadata <- read.dcf("inst/cuda-ml-backend.dcf")[1L, , drop = TRUE]
stopifnot(
  identical(names(versions), c("nvforest", "treelite")),
  identical(
    package_version(versions[["nvforest"]]),
    package_version(metadata[["nvForest"]])
  ),
  identical(
    package_version(versions[["treelite"]]),
    package_version(metadata[["Treelite"]])
  )
)

dynamic_symbols <- system2(
  "nm",
  c("-D", "--defined-only", shQuote(backend)),
  stdout = TRUE,
  stderr = TRUE
)
stopifnot(is.null(attr(dynamic_symbols, "status")))
dynamic_symbol_names <- sub("^.*[[:space:]]", "", dynamic_symbols)
treelite_symbols <- grepl(
  paste0(
    "^(Treelite|TREELITE_|_Z[0-9]+Treelite|_ZNK?8treelite|",
    "_ZT[ISV]N8treelite|_Z(GVZN|THN|TWN)8treelite|",
    "_ZT[hv].*N8treelite)"
  ),
  dynamic_symbol_names
)
stopifnot(!any(treelite_symbols))

version_info <- system2(
  "readelf",
  c("--version-info", shQuote(backend)),
  stdout = TRUE,
  stderr = TRUE
)
stopifnot(is.null(attr(version_info, "status")))
maximum_version <- function(prefix) {
  pattern <- paste0(prefix, "_([0-9]+(?:[.][0-9]+)+)")
  matches <- regmatches(
    version_info,
    gregexpr(pattern, version_info, perl = TRUE)
  )
  values <- unique(sub(paste0("^", prefix, "_"), "", unlist(matches)))
  stopifnot(length(values) > 0L)
  max(package_version(values))
}
stopifnot(
  maximum_version("GLIBC") <= package_version("2.28"),
  maximum_version("GLIBCXX") <= package_version("3.4.25"),
  maximum_version("CXXABI") <= package_version("1.3.11")
)

message("CPU-only nvForest backend audit passed")
