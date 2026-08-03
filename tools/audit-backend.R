args <- commandArgs(trailingOnly = TRUE)
stopifnot(
  length(args) == 2L,
  file.exists(args[[1L]]),
  file.exists(args[[2L]])
)

backend <- normalizePath(args[[1L]], mustWork = TRUE)
cuobjdump <- normalizePath(args[[2L]], mustWork = TRUE)

library(Rcpp)
stopifnot(!"cuda.ml" %in% names(getLoadedDLLs()))
dll <- dyn.load(backend, local = FALSE, now = TRUE)
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
  file.path("inst", "native-symbols.txt"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
stopifnot(identical(registered_manifest, expected_manifest))

versions <- .Call("_cuda_ml_backend_versions", PACKAGE = "cuda.ml")
metadata_path <- file.path("inst", "cuda-ml-backend.dcf")
stopifnot(file.exists(metadata_path))
metadata <- read.dcf(metadata_path)
stopifnot(
  nrow(metadata) == 1L,
  all(
    c(
      "CUDA",
      "RAPIDS",
      "nvForest",
      "Treelite",
      "Architectures"
    ) %in%
      colnames(metadata)
  )
)
metadata <- metadata[1L, ]

cuda_version <- metadata[["CUDA"]]
stopifnot(grepl("^[0-9]+[.][0-9]+[.][0-9]+$", cuda_version))
cuda <- as.integer(strsplit(cuda_version, ".", fixed = TRUE)[[1L]])
stopifnot(length(cuda) == 3L, !anyNA(cuda))
expected_cudart <- cuda[[1L]] * 1000L + cuda[[2L]] * 10L
expected_cudart_soname <- paste0("libcudart.so.", cuda[[1L]])

architectures <- strsplit(
  metadata[["Architectures"]],
  ";",
  fixed = TRUE
)[[1L]]
architecture_matches <- regexec(
  "^([0-9]+)-(real|virtual)$",
  architectures
)
architecture_parts <- regmatches(architectures, architecture_matches)
stopifnot(
  length(architectures) > 0L,
  all(lengths(architecture_parts) == 3L),
  !anyDuplicated(architectures)
)
architecture_values <- vapply(
  architecture_parts,
  `[[`,
  character(1),
  2L
)
architecture_kinds <- vapply(
  architecture_parts,
  `[[`,
  character(1),
  3L
)
expected_sass <- paste0(
  "sm_",
  architecture_values[architecture_kinds == "real"]
)
expected_virtual <- paste0(
  "sm_",
  architecture_values[architecture_kinds == "virtual"]
)
stopifnot(
  identical(
    package_version(versions[["cuml"]]),
    package_version(metadata[["RAPIDS"]])
  ),
  identical(
    package_version(versions[["nvforest"]]),
    package_version(metadata[["nvForest"]])
  ),
  identical(
    package_version(versions[["treelite"]]),
    package_version(metadata[["Treelite"]])
  ),
  identical(versions[["cuda_runtime"]], expected_cudart)
)

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
  setequal(sass, expected_sass),
  setequal(virtual, expected_virtual)
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
    c(
      "libcuml.so",
      "libcublas.so.13",
      "libcusolver.so.12",
      expected_cudart_soname,
      "libcusparse.so.12",
      "libnvforest++.so"
    )
  )
)

dynamic_symbols <- system2(
  "nm",
  c("-D", "--defined-only", shQuote(backend)),
  stdout = TRUE,
  stderr = TRUE
)
dynamic_symbol_names <- sub("^.*[[:space:]]", "", dynamic_symbols)
treelite_symbols <- grepl(
  paste0(
    "^(Treelite|TREELITE_|_Z[0-9]+Treelite|_ZNK?8treelite|",
    "_ZT[ISV]N8treelite|_Z(GVZN|THN|TWN)8treelite|",
    "_ZT[hv].*N8treelite)"
  ),
  dynamic_symbol_names
)
stopifnot(
  is.null(attr(dynamic_symbols, "status")),
  !any(treelite_symbols)
)

versions <- system2(
  "readelf",
  c("--version-info", shQuote(backend)),
  stdout = TRUE,
  stderr = TRUE
)
stopifnot(is.null(attr(versions, "status")))
maximum_version <- function(prefix) {
  pattern <- paste0(prefix, "_([0-9]+(?:[.][0-9]+)+)")
  matches <- regmatches(versions, gregexpr(pattern, versions, perl = TRUE))
  values <- unique(sub(paste0("^", prefix, "_"), "", unlist(matches)))
  stopifnot(length(values) > 0L)
  max(package_version(values))
}
stopifnot(
  maximum_version("GLIBC") <= package_version("2.28"),
  maximum_version("GLIBCXX") <= package_version("3.4.25"),
  maximum_version("CXXABI") <= package_version("1.3.11")
)

message("cuda.ml backend audit passed")
