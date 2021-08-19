decision_tree_match_split_criterion <- function(criterion = c("gini", "entropy", "mse", "mae"),
                                                classification = TRUE) {
  criterion <- criterion %||% ifelse(classification, "gini", "mse")
  criterion <- match.arg(criterion)

  if (classification && criterion %in% c("mse", "mae")) {
    stop("'", criterion, "' is not a valid criterion for classification.")
  }

  switch(criterion,
    gini = 0L,
    entropy = 1L,
    mse = 2L,
    mae = 3L
  )
}
