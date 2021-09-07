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
      list(class = c(cls, "cuml_model"), mode = mode, xptr = xptr),
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

cuml_is_classifier <- function(model) {
  identical(model$mode, "classifier")
}

#' Determine whether a CuML model can predict class probabilities.
#'
#' Given a trained CuML model, return \code{TRUE} if the model is a classifier
#' and is capable of outputting class probabilities as prediction results (e.g.,
#' if the model is a KNN or an ensemble classifier), otherwise return
#' \code{FALSE}.
#'
#' @param model A trained CuML model.
cuml_can_predict_class_probabilities <- function(model) {
  UseMethod("cuml_can_predict_class_probabilities")
}

cuml_can_predict_class_probabilities.default <- function(model) {
  report_undefined_fn("cuml_can_predict_class_probabilities", model)
}

cuml_can_predict_class_probabilities.cuml_model <- function(model) {
  FALSE
}

cuml_can_predict_class_probabilities.cuml_fil <- cuml_is_classifier

cuml_can_predict_class_probabilities.cuml_knn <- cuml_is_classifier

cuml_can_predict_class_probabilities.cuml_rand_forest <- cuml_is_classifier

#' Make predictions on new data points.
#'
#' Use a trained CuML model to make predictions on new data points.
#' Notice calling \code{cuml_predict()} will be identical to calling the
#' \code{predict()} S3 generic, except for \code{cuml_predict()} also comes
#' with proper documentation on all possible predict options (such as
#' \code{output_class_probabilities}) and will emit a sensible error message
#' when a predict option is not applicable for a given model.
#'
#' @param model A trained CuML model.
#' @param x A matrix or dataframe containing new data points.
#' @param output_class_probabilities Whetoer to output class probabilities.
#'   Setting \code{output_class_probabilities} to \code{TRUE} is only valid
#'   when the model being applied is a classification model and supports class
#'   probabilities output. CuML classification models that support class
#'   probabilities include \code{knn}, \code{fil}, and \code{rand_forest}.
#'
#' @export
cuml_predict <- function(model, x, output_class_probabilities = NULL, ...) {
  # TODO:
}
