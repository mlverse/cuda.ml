match_split_criterion <- function(criterion = c("gini", "entropy", "mse", "mae"), mode = "classification") {
  criterion <- match.arg(criterion)

  if (identical(mode, "classification") && criterion %in% c("mse", "mae")) {
    stop("'", criterion, "' is not a valid criterion for classification.")
  }

  switch(criterion,
    gini = 0L,
    entropy = 1L,
    mse = 2L,
    mae = 3L
  )
}
