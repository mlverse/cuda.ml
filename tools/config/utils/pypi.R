# Resolve the full dependency tree for a PyPI package and return download URLs
# for all packages (including transitive deps) that contain C++ headers.
#
# This is used to download libcuml and all its header-only dependencies
# (libraft, librmm, rapids-logger, nvidia-cccl, etc.) from PyPI without
# needing pip installed.

pypi_package_info <- function(package, version = NULL) {
  url <- if (is.null(version)) {
    sprintf("https://pypi.org/pypi/%s/json", package)
  } else {
    sprintf("https://pypi.org/pypi/%s/%s/json", package, version)
  }
  tmp <- tempfile(fileext = ".json")
  download.file(url, tmp, quiet = TRUE)
  jsonlite::fromJSON(tmp)
}

pypi_wheel_url <- function(package, version = NULL, platform = "x86_64") {
  info <- pypi_package_info(package, version)
  urls <- info$urls
  # Find a matching wheel for the platform
  idx <- grep(platform, urls$filename)
  if (length(idx) == 0) {
    # Try platform-independent wheels
    idx <- grep("none-any", urls$filename)
  }
  if (length(idx) == 0) {
    stop(sprintf("No wheel found for %s (platform: %s)", package, platform))
  }
  list(
    url = urls$url[idx[1]],
    filename = urls$filename[idx[1]],
    version = info$info$version,
    requires_dist = info$info$requires_dist
  )
}

# Parse a PEP 508 dependency string into package name
# e.g. "libraft-cu12==26.4.*" -> "libraft-cu12"
# e.g. "numpy>=1.0; extra == 'test'" -> "numpy" (but we skip extras)
parse_dep_name <- function(dep_str) {
  # Skip deps with extras/markers like "; extra == ..."
  if (grepl("; extra\\s*==", dep_str)) return(NULL)
  # Extract package name (everything before version specifier or semicolon)
  gsub("[\\s;(<>=!\\[].*", "", dep_str, perl = TRUE)
}

# Resolve all transitive dependencies that look like C++ library packages
# (lib*, rapids-*, nvidia-cccl-*, nvidia-nvjitlink-*)
resolve_native_deps <- function(package, version = NULL, seen = character()) {
  if (package %in% seen) return(list())
  seen <- c(seen, package)

  info <- tryCatch(
    pypi_wheel_url(package, version),
    error = function(e) NULL
  )
  if (is.null(info)) return(list())

  result <- list()
  result[[package]] <- info$url

  # Only chase transitive deps for native/C++ packages
  if (!is.null(info$requires_dist)) {
    for (dep in info$requires_dist) {
      dep_name <- parse_dep_name(dep)
      if (is.null(dep_name)) next
      # Only follow native library deps (lib*, rapids-*, nvidia-cccl*, nvidia-nvjitlink*)
      if (grepl("^(lib|rapids-|nvidia-cccl|nvidia-nvjitlink)", dep_name)) {
        sub_deps <- resolve_native_deps(dep_name, seen = seen)
        seen <- c(seen, names(sub_deps))
        result <- c(result, sub_deps)
      }
    }
  }

  result
}
