#' Model with numeric input.
#'
#' Generic ML model with one numeric input provided by a matrix or dataframe.
#'
#' @param x The input matrix or dataframe. Each data point should be a row
#'   and should consist of numeric values only.
#'
#' @name model-with-numeric-input
NULL

#' Supervised ML model with numeric output.
#'
#' Supervised ML model outputting a numeric value for each train/test sample.
#'
#' @param y A numeric vector of desired responses.
#'
#' @name supervised-model-with-numeric-output
NULL

#' Supervised ML algorithm with formula support.
#'
#' Supervised ML algorithm that supports specifying predictor(s) and response
#' variable using the formula syntax.
#'
#' @param formula If 'x' is a dataframe, then a R formula syntax of the form
#'   '<response col> ~ .' or
#'   '<response col> ~ <predictor 1> + <predictor 2> + ...'
#'   may be used to specify the response column and the predictor column(s).
#'
#' @name supervised-model-formula-spec
NULL

#' Supervised ML model with "classification" or "regression" modes.
#'
#' Supervised ML model that needs to be trained with different cuML routines
#' depending on whether "classification" or "regression" mode is required.
#'
#' @param mode Type of task to perform. Should be either "classification" or
#'   "regression".
#'
#' @name supervised-model-classification-or-regression-mode
NULL
