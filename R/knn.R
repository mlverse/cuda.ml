knn_match_algo <- function(algo = c("brute", "ivfflat", "ivfpq", "ivfsq")) {
  algo <- match.arg(algo)

  switch(algo,
    brute = 0L,
    ivfflat = 1L,
    ivfpq = 2L,
    ivfsq = 3L
  )
}

knn_match_metric <- function(metric = c("euclidean", "l2", "l1", "cityblock", "taxicab", "manhattan", "braycurtis", "canberra", "minkowski", "lp", "chebyshev", "linf", "jensenshannon", "cosine", "correlation")) {
  metric <- match.arg(metric)

  switch(metric,
    euclidean = 1L,
    l2 = 1L,
    l1 = 3L,
    cityblock = 3L,
    taxicab = 3L,
    manhattan = 3L,
    braycurtis = 14L,
    canberra = 8L,
    minkowski = 9L,
    lp = 9L,
    chebyshev = 7L,
    linf = 7L,
    jensenshannon = 15L,
    cosine = 2L,
    correlation = 10L
  )
}

#' Build a specification for the "ivfflat" KNN query algorithm.
#'
#' Build a specification of the flat-inverted-file KNN query algorithm, with all
#' required parameters specified explicitly.
#'
#' @template knn-algo-common
#'
#' @return An object encapsulating all required parameters of the "ivfflat" KNN
#'   query algorithm.
#'
#' @export
cuda_ml_knn_algo_ivfflat <- function(nlist, nprobe) {
  list(
    type = 1L,
    params = list(
      nlist = as.integer(nlist),
      nprobe = as.integer(nprobe)
    )
  )
}

#' Build a specification for the "ivfpq" KNN query algorithm.
#'
#' Build a specification of the inverted-file-product-quantization KNN query
#' algorithm, with all required parameters specified explicitly.
#'
#' @template knn-algo-common
#' @template knn-algo-ivfpq
#'
#' @return An object encapsulating all required parameters of the "ivfpq" KNN
#'   query algorithm.
#'
#' @export
cuda_ml_knn_algo_ivfpq <- function(nlist, nprobe, m, n_bits,
                                   use_precomputed_tables = FALSE) {
  list(
    type = 2L,
    params = list(
      nlist = as.integer(nlist),
      nprobe = as.integer(nprobe),
      M = as.integer(m),
      usePrecomputedTables = as.logical(use_precomputed_tables)
    )
  )
}

#' Build a specification for the "ivfsq" KNN query algorithm.
#'
#' Build a specification of the inverted-file-scalar-quantization KNN query
#' algorithm, with all required parameters specified explicitly.
#'
#' @template knn-algo-common
#' @template knn-algo-ivfsq
#'
#' @return An object encapsulating all required parameters of the "ivfsq" KNN
#'   query algorithm.
#'
#' @export
cuda_ml_knn_algo_ivfsq <- function(nlist, nprobe,
                                   qtype = c("QT_8bit", "QT_4bit", "QT_8bit_uniform", "QT_4bit_uniform", "QT_fp16", "QT_8bit_direct", "QT_6bit"),
                                   encode_residual = FALSE) {
  list(
    type = 3L,
    params = list(
      nlist = as.integer(nlist),
      nprobe = as.integer(nprobe),
      qtype = match.arg(qtype),
      encodeResidual = as.logical(encode_residual)
    )
  )
}

#' Build a KNN model.
#'
#' Build a k-nearest-model for classification or regression tasks.
#'
#' @template supervised-model-inputs
#' @template supervised-model-output
#' @template ellipsis-unused
#' @param algo The query algorithm to use. Must be one of
#'   {"brute", "ivfflat", "ivfpq", "ivfsq"} or a KNN algorithm specification
#'   constructed using the \code{cuda_ml_knn_algo_*} family of functions.
#'   If the algorithm is specified by one of the \code{cuda_ml_knn_algo_*}
#'   functions, then values of all required parameters of the algorithm will
#'   need to be specified explicitly.
#'   If the algorithm is specified by a character vector, then parameters for
#'   the algorithm are generated automatically.
#'
#'   Descriptions of supported algorithms:
#'     - "brute": for brute-force, slow but produces exact results.
#'     - "ivfflat": for inverted file, divide the dataset in partitions
#'                  and perform search on relevant partitions only.
#'     - "ivfpq": for inverted file and product quantization (vectors
#'                are divided into sub-vectors, and each sub-vector is encoded
#'                using intermediary k-means clusterings to provide partial
#'                information).
#'     - "ivfsq": for inverted file and scalar quantization (vectors components
#'                are quantized into reduced binary representation allowing
#'                faster distances calculations).
#'
#'   Default: "brute".
#' @param metric Distance metric to use. Must be one of {"euclidean", "l2",
#'   "l1", "cityblock", "taxicab", "manhattan", "braycurtis", "canberra",
#'   "minkowski", "lp", "chebyshev", "linf", "jensenshannon", "cosine",
#'   "correlation"}.
#'   Default: "euclidean".
#' @param p Parameter for the Minkowski metric. If p = 1, then the metric is
#'   equivalent to manhattan distance (l1). If p = 2, the metric is equivalent
#'   to euclidean distance (l2).
#' @param neighbors Number of nearest neighbors to query. Default: 5L.
#'
#' @return A KNN model that can be used with the 'predict' S3 generic to make
#'   predictions on new data points.
#'   The model object contains the following:
#'     - "knn_index": a GPU pointer to the KNN index.
#'     - "algo": enum value of the algorithm being used for the KNN query.
#'     - "metric": enum value of the distance metric used in KNN computations.
#'     - "p": parameter for the Minkowski metric.
#'     - "n_samples": number of input data points.
#'     - "n_dims": dimension of each input data point.
#'
#' @examples
#'
#' library(cuda.ml)
#' library(MASS)
#' library(magrittr)
#' library(purrr)
#'
#' set.seed(0L)
#'
#' centers <- list(c(3, 3), c(-3, -3), c(-3, 3))
#'
#' gen_pts <- function(cluster_sz) {
#'   pts <- centers %>%
#'     map(~ mvrnorm(cluster_sz, mu = .x, Sigma = diag(2)))
#'
#'   rlang::exec(rbind, !!!pts) %>% as.matrix()
#' }
#'
#' gen_labels <- function(cluster_sz) {
#'   seq_along(centers) %>%
#'     sapply(function(x) rep(x, cluster_sz)) %>%
#'     factor()
#' }
#'
#' sample_cluster_sz <- 1000
#' sample_pts <- cbind(
#'   gen_pts(sample_cluster_sz) %>% as.data.frame(),
#'   label = gen_labels(sample_cluster_sz)
#' )
#'
#' model <- cuda_ml_knn(label ~ ., sample_pts, algo = "ivfflat", metric = "euclidean")
#'
#' test_cluster_sz <- 10
#' test_pts <- gen_pts(test_cluster_sz) %>% as.data.frame()
#'
#' predictions <- predict(model, test_pts)
#' print(predictions, n = 30)
#' @importFrom ellipsis check_dots_used
#' @export
cuda_ml_knn <- function(x, ...) {
  check_dots_used()
  UseMethod("cuda_ml_knn")
}

#' @rdname cuda_ml_knn
#' @export
cuda_ml_knn.default <- function(x, ...) {
  report_undefined_fn("cuda_ml_knn", x)
}

#' @rdname cuda_ml_knn
#' @export
cuda_ml_knn.data.frame <- function(x, y,
                                   algo = c("brute", "ivfflat", "ivfpq", "ivfsq"),
                                   metric = c("euclidean", "l2", "l1", "cityblock", "taxicab", "manhattan", "braycurtis", "canberra", "minkowski", "chebyshev", "jensenshannon", "cosine", "correlation"),
                                   p = 2.0,
                                   neighbors = 5L,
                                   ...) {
  processed <- hardhat::mold(x, y)

  cuda_ml_knn_bridge(
    processed = processed,
    algo = algo,
    metric = metric,
    p = p,
    neighbors = neighbors
  )
}

#' @rdname cuda_ml_knn
#' @export
cuda_ml_knn.matrix <- function(x, y,
                               algo = c("brute", "ivfflat", "ivfpq", "ivfsq"),
                               metric = c("euclidean", "l2", "l1", "cityblock", "taxicab", "manhattan", "braycurtis", "canberra", "minkowski", "chebyshev", "jensenshannon", "cosine", "correlation"),
                               p = 2.0,
                               neighbors = 5L,
                               ...) {
  processed <- hardhat::mold(x, y)

  cuda_ml_knn_bridge(
    processed = processed,
    algo = algo,
    metric = metric,
    p = p,
    neighbors = neighbors
  )
}

#' @rdname cuda_ml_knn
#' @export
cuda_ml_knn.formula <- function(formula, data,
                                algo = c("brute", "ivfflat", "ivfpq", "ivfsq"),
                                metric = c("euclidean", "l2", "l1", "cityblock", "taxicab", "manhattan", "braycurtis", "canberra", "minkowski", "chebyshev", "jensenshannon", "cosine", "correlation"),
                                p = 2.0,
                                neighbors = 5L,
                                ...) {
  processed <- hardhat::mold(formula, data)

  cuda_ml_knn_bridge(
    processed = processed,
    algo = algo,
    metric = metric,
    p = p,
    neighbors = neighbors
  )
}

#' @rdname cuda_ml_knn
#' @export
cuda_ml_knn.recipe <- function(x, data,
                               algo = c("brute", "ivfflat", "ivfpq", "ivfsq"),
                               metric = c("euclidean", "l2", "l1", "cityblock", "taxicab", "manhattan", "braycurtis", "canberra", "minkowski", "chebyshev", "jensenshannon", "cosine", "correlation"),
                               p = 2.0,
                               neighbors = 5L,
                               ...) {
  processed <- hardhat::mold(x, data)

  cuda_ml_knn_bridge(
    processed = processed,
    algo = algo,
    metric = metric,
    p = p,
    neighbors = neighbors
  )
}

cuda_ml_knn_bridge <- function(processed, algo, metric, p, neighbors) {
  hardhat::validate_predictors_are_numeric(processed$predictors)
  hardhat::validate_outcomes_are_univariate(processed$outcomes)
  x <- as.matrix(processed$predictors)
  y <- processed$outcomes[[1]]

  if (is.character(algo)) {
    algo_type <- knn_match_algo(algo)
    algo_params <- list()
  } else {
    algo_type <- algo$type
    algo_params <- algo$params
  }
  metric <- knn_match_metric(metric)

  if (is.factor(y)) {
    # classification
    prediction_mode <- "classification"
    model_xptr <- .knn_classifier_fit(
      x = x,
      y = as.integer(y),
      algo = algo_type,
      metric = metric,
      p = as.numeric(p),
      algo_params = algo_params
    )
  } else {
    prediction_mode <- "regression"
    model_xptr <- .knn_regressor_fit(
      x = x,
      y = as.numeric(y),
      algo = algo_type,
      metric = metric,
      p = as.numeric(p),
      algo_params = algo_params
    )
  }

  new_model(
    cls = "cuda_ml_knn",
    mode = prediction_mode,
    xptr = model_xptr,
    neighbors = as.integer(neighbors),
    blueprint = processed$blueprint
  )
}

#' Make predictions on new data points.
#'
#' Make predictions on new data points using a CuML KNN model.
#'
#' @template predict
#' @template output-class-probabilities
#'
#' @importFrom ellipsis check_dots_used
#' @export
predict.cuda_ml_knn <- function(object, x, output_class_probabilities = NULL, ...) {
  check_dots_used()

  processed <- hardhat::forge(x, object$blueprint)

  predict_cuda_ml_knn_bridge(
    model = object,
    processed = processed,
    output_class_probabilities = output_class_probabilities
  )
}

predict_cuda_ml_knn_bridge <- function(model, processed, output_class_probabilities) {
  out <- switch(model$mode,
    classification = {
      predict_cuda_ml_knn_classification_impl(
        model = model,
        processed = processed,
        output_class_probabilities = output_class_probabilities %||% FALSE
      )
    },
    regression = {
      if (!is.null(output_class_probabilities)) {
        stop("'output_class_probabilities' is not applicable for regression tasks!")
      }

      predict_cuda_ml_knn_regression_impl(
        model = model, processed = processed
      )
    }
  )
  hardhat::validate_prediction_size(out, processed$predictors)

  out
}

predict_cuda_ml_knn_classification_impl <- function(model, processed, output_class_probabilities) {
  if (output_class_probabilities) {
    preds <- .knn_classifier_predict_probabilities(
      model = model$xptr,
      x = as.matrix(processed$predictors),
      n_neighbors = model$neighbors
    )

    postprocess_class_probabilities(preds, model)
  } else {
    preds <- .knn_classifier_predict(
      model = model$xptr,
      x = as.matrix(processed$predictors),
      n_neighbors = model$neighbors
    )

    postprocess_classification_results(preds, model)
  }
}

predict_cuda_ml_knn_regression_impl <- function(model, processed) {
  preds <- .knn_regressor_predict(
    model = model$xptr,
    x = as.matrix(processed$predictors),
    n_neighbors = model$neighbors
  )

  postprocess_regression_results(preds)
}

# register the CuML-based knn model for parsnip
register_knn_model <- function(pkgname) {
  for (mode in c("classification", "regression")) {
    parsnip::set_model_engine(
      model = "nearest_neighbor", mode = mode, eng = pkgname
    )
  }

  parsnip::set_dependency(model = "nearest_neighbor", eng = pkgname, pkg = pkgname)

  parsnip::set_model_arg(
    model = "nearest_neighbor",
    eng = pkgname,
    parsnip = "neighbors",
    original = "neighbors",
    func = list(pkg = "dials", fun = "neighbors", range = c(1, 15)),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    model = "nearest_neighbor",
    eng = pkgname,
    parsnip = "dist_power",
    original = "p",
    func = list(pkg = "dials", fun = "dist_power", range = c(1 / 10, 2)),
    has_submodel = FALSE
  )

  for (mode in c("classification", "regression")) {
    parsnip::set_fit(
      model = "nearest_neighbor",
      eng = pkgname,
      mode = mode,
      value = list(
        interface = "formula",
        protect = c("formula", "data"),
        func = c(pkg = pkgname, fun = "cuda_ml_knn"),
        defaults = list(algo = "ivfflat", metric = "euclidean")
      )
    )

    parsnip::set_encoding(
      model = "nearest_neighbor",
      eng = pkgname,
      mode = mode,
      options = list(
        predictor_indicators = "none",
        compute_intercept = FALSE,
        remove_intercept = FALSE,
        allow_sparse_x = TRUE
      )
    )
  }

  for (type in c("class", "prob")) {
    parsnip::set_pred(
      model = "nearest_neighbor",
      eng = pkgname,
      mode = "classification",
      type = type,
      value = list(
        pre = NULL,
        post = NULL,
        func = c(fun = "predict"),
        args = list(
          quote(object$fit),
          quote(new_data),
          identical(type, "prob") # output_class_probabilities
        )
      )
    )
  }

  parsnip::set_pred(
    model = "nearest_neighbor",
    eng = pkgname,
    mode = "regression",
    type = "numeric",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        quote(object$fit),
        quote(new_data)
      )
    )
  )
}
