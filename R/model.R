#' @importFrom stats predict
NULL

match_eig_algo <- function(eig_algo = c("dq", "jacobi")) {
  eig_algo <- match.arg(eig_algo)

  switch(eig_algo,
    dq = 0L,
    jacobi = 1L
  )
}

new_model <- function(cls,
                      mode = c("classification", "regression"),
                      xptr = NULL, ...) {
  mode <- match.arg(mode)
  do.call(
    hardhat::new_model,
    c(
      list(class = cls, mode = mode, xptr = xptr),
      rlang::dots_list(...)
    )
  )
}

get_pred_levels <- function(model) {
  levels(model$blueprint$ptypes$outcomes[[1]])
}

postprocess_classification_results <- function(predictions, model) {
  predictions <- as.integer(predictions)
  pred_levels <- get_pred_levels(model)
  predictions <- pred_levels[predictions]
  predictions <- factor(predictions, levels = pred_levels)

  hardhat::spruce_class(predictions)
}

postprocess_class_probabilities <- function(predictions, model) {
  pred_levels <- get_pred_levels(model)

  hardhat::spruce_prob(pred_levels, predictions)
}

postprocess_regression_results <- function(predictions) {
  hardhat::spruce_numeric(predictions)
}

report_undefined_fn <- function(fn_name, x) {
  stop(
    "`", fn_name, "()` is undefined for object of class ",
    paste(class(x), sep = " "),
    call. = FALSE
  )
}

#' Transform data using a trained cuML model.
#'
#' Given a trained cuML model, transform an input dataset using that model.
#'
#' @template cuml-transform
#'
#' @importFrom ellipsis check_dots_used
#' @export
cuml_transform <- function(model, x, ...) {
  check_dots_used()
  UseMethod("cuml_transform")
}


#' Apply the inverse transformation defined by a trained cuML model.
#'
#' Given a trained cuML model, apply the inverse transformation defined by that
#' model to an input dataset.
#'
#' @template cuml-transform
#'
#' @importFrom ellipsis check_dots_used
#' @export
cuml_inverse_transform <- function(model, x, ...) {
  check_dots_used()
  UseMethod("cuml_inverse_transform")
}
