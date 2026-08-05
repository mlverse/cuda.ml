# Compare the working tree's exported function and registered S3 method
# signatures with a released git ref without loading either package version.
cli_args <- commandArgs(trailingOnly = TRUE)
stopifnot(
  "Usage: Rscript tools/diff-api.R <baseline-ref>" = length(cli_args) == 1L
)

baseline <- cli_args[[1L]]

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

git_lines <- function(...) {
  result <- system2("git", c(...), stdout = TRUE, stderr = TRUE)
  status <- attr(result, "status") %||% 0L
  if (status != 0L) {
    stop(paste(result, collapse = "\n"), call. = FALSE)
  }
  result
}

namespace_exports <- function(lines) {
  matches <- regexec("^export\\(([^)]+)\\)$", lines)
  captures <- regmatches(lines, matches)
  sort(vapply(captures[lengths(captures) == 2L], `[[`, character(1), 2L))
}

namespace_methods <- function(lines) {
  matches <- regexec("^S3method\\(([^,]+),([^)]+)\\)$", lines)
  captures <- regmatches(lines, matches)
  captures <- captures[lengths(captures) == 3L]
  sort(vapply(
    captures,
    function(capture) {
      generic <- sub("^.*::", "", capture[[2L]])
      paste(generic, capture[[3L]], sep = ".")
    },
    character(1)
  ))
}

source_functions <- function(sources) {
  functions <- new.env(parent = emptyenv())
  aliases <- list()

  for (source in sources) {
    expressions <- parse(text = source)
    for (expression in expressions) {
      if (
        !is.call(expression) ||
          !identical(expression[[1L]], quote(`<-`)) ||
          !is.symbol(expression[[2L]])
      ) {
        next
      }

      name <- as.character(expression[[2L]])
      value <- expression[[3L]]
      if (is.call(value) && identical(value[[1L]], quote(`function`))) {
        assign(name, eval(value, envir = baseenv()), envir = functions)
      } else if (is.symbol(value)) {
        aliases[[name]] <- as.character(value)
      }
    }
  }

  repeat {
    unresolved <- names(aliases)
    for (name in unresolved) {
      target <- aliases[[name]]
      if (exists(target, envir = functions, inherits = FALSE)) {
        assign(name, get(target, envir = functions), envir = functions)
        aliases[[name]] <- NULL
      }
    }
    if (length(aliases) == length(unresolved)) {
      break
    }
  }

  functions
}

ref_sources <- function(ref) {
  paths <- git_lines("ls-tree", "-r", "--name-only", ref, "R")
  paths <- paths[grepl("[.]R$", paths)]
  lapply(paths, function(path) git_lines("show", paste0(ref, ":", path)))
}

worktree_sources <- function() {
  paths <- list.files("R", pattern = "[.]R$", full.names = TRUE)
  lapply(paths, readLines, warn = FALSE)
}

signature <- function(functions, name) {
  if (!exists(name, envir = functions, inherits = FALSE)) {
    return("<definition not found>")
  }
  paste(capture.output(args(get(name, envir = functions))), collapse = " ")
}

old_namespace <- git_lines("show", paste0(baseline, ":NAMESPACE"))
new_namespace <- readLines("NAMESPACE", warn = FALSE)
old_exports <- namespace_exports(old_namespace)
new_exports <- namespace_exports(new_namespace)
old_functions <- source_functions(ref_sources(baseline))
new_functions <- source_functions(worktree_sources())

removed <- setdiff(old_exports, new_exports)
added <- setdiff(new_exports, old_exports)
common <- intersect(old_exports, new_exports)
changed <- common[
  vapply(
    common,
    function(name) {
      !identical(
        signature(old_functions, name),
        signature(new_functions, name)
      )
    },
    logical(1)
  )
]

writeLines(c("Removed exports:", paste0("- ", removed), ""))
writeLines(c("Added exports:", paste0("- ", added), ""))
writeLines("Changed signatures:")
for (name in changed) {
  writeLines(c(
    paste0("- ", name),
    paste0("  old: ", signature(old_functions, name)),
    paste0("  new: ", signature(new_functions, name))
  ))
}

old_methods <- namespace_methods(old_namespace)
new_methods <- namespace_methods(new_namespace)
removed_methods <- setdiff(old_methods, new_methods)
added_methods <- setdiff(new_methods, old_methods)
common_methods <- intersect(old_methods, new_methods)
changed_methods <- common_methods[
  vapply(
    common_methods,
    function(name) {
      !identical(
        signature(old_functions, name),
        signature(new_functions, name)
      )
    },
    logical(1)
  )
]

writeLines(c("", "Removed registered S3 methods:", paste0("- ", removed_methods)))
writeLines(c("", "Added registered S3 methods:", paste0("- ", added_methods)))
writeLines(c("", "Changed registered S3 method signatures:"))
for (name in changed_methods) {
  writeLines(c(
    paste0("- ", name),
    paste0("  old: ", signature(old_functions, name)),
    paste0("  new: ", signature(new_functions, name))
  ))
}
