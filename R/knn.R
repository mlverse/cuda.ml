knn_match_algo <- function(algo) {
  algo <- match.arg(algo, c("brute", "ivfflat", "ivfpq"))

  switch(algo, brute = 0L, ivfflat = 1L, ivfpq = 2L)
}

knn_match_metric <- function(metric) {
  metric <- match.arg(
    metric,
    c(
      "euclidean",
      "l2",
      "l1",
      "cityblock",
      "taxicab",
      "manhattan",
      "braycurtis",
      "canberra",
      "minkowski",
      "lp",
      "chebyshev",
      "linf",
      "jensenshannon",
      "cosine",
      "correlation"
    )
  )

  switch(
    metric,
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

#' Configure an approximate KNN query algorithm
#'
#' For the main path, pass \code{"ivfflat"} or \code{"ivfpq"} directly to the
#' \code{algo} argument of \code{cuda_ml_knn()}; cuda.ml then lets the backend
#' choose the index parameters. Use these constructors only when those
#' parameters need to be set explicitly.
#'
#' Both algorithms partition the training data into \code{nlist} cells and
#' search \code{nprobe} cells for each query. IVFFlat stores the original
#' vectors and therefore needs only those two parameters. IVFPQ also compresses
#' vectors using product quantization, so it additionally requires the number
#' of subquantizers (\code{m}) and the bits allocated to each subquantizer
#' (\code{n_bits}). The distinct constructors keep the required parameters for
#' each algorithm explicit.
#'
#' @template knn-algo-common
#' @template knn-algo-ivfpq
#'
#' @return A KNN algorithm specification to pass to the \code{algo} argument of
#'   \code{cuda_ml_knn()}.
#'
#' @name cuda_ml_knn_algo
#' @seealso \code{\link{cuda_ml_knn}()}
#' @export
cuda_ml_knn_algo_ivfflat <- function(nlist, nprobe) {
  stopifnot(
    "`nlist` must be one positive whole number" = is.numeric(nlist) &&
      length(nlist) == 1L &&
      is.finite(nlist) &&
      nlist >= 1L &&
      nlist == as.integer(nlist),
    "`nprobe` must be one positive whole number no greater than `nlist`" = is.numeric(
      nprobe
    ) &&
      length(nprobe) == 1L &&
      is.finite(nprobe) &&
      nprobe >= 1L &&
      nprobe == as.integer(nprobe) &&
      nprobe <= nlist
  )
  structure(
    list(
      type = 1L,
      params = list(
        nlist = as.integer(nlist),
        nprobe = as.integer(nprobe)
      )
    ),
    class = "cuda_ml_knn_algo"
  )
}

knn_validate_ivfpq_params <- function(
  nlist,
  nprobe,
  m,
  n_bits
) {
  stopifnot(
    "`nlist` must be one positive whole number" = is.numeric(nlist) &&
      length(nlist) == 1L &&
      is.finite(nlist) &&
      nlist >= 1L &&
      nlist == as.integer(nlist),
    "`nprobe` must be one positive whole number no greater than `nlist`" = is.numeric(
      nprobe
    ) &&
      length(nprobe) == 1L &&
      is.finite(nprobe) &&
      nprobe >= 1L &&
      nprobe == as.integer(nprobe) &&
      nprobe <= nlist,
    "`m` must be one positive whole number" = is.numeric(m) &&
      length(m) == 1L &&
      is.finite(m) &&
      m >= 1L &&
      m == as.integer(m),
    "`n_bits` must be one whole number between 4 and 8" = is.numeric(n_bits) &&
      length(n_bits) == 1L &&
      is.finite(n_bits) &&
      n_bits >= 4L &&
      n_bits <= 8L &&
      n_bits == as.integer(n_bits),
    "`m * n_bits` must be divisible by 8" = (m * n_bits) %% 8L == 0L
  )
  invisible(TRUE)
}

#' @rdname cuda_ml_knn_algo
#' @export
cuda_ml_knn_algo_ivfpq <- function(
  nlist,
  nprobe,
  m,
  n_bits
) {
  knn_validate_ivfpq_params(
    nlist,
    nprobe,
    m,
    n_bits
  )
  structure(
    list(
      type = 2L,
      params = list(
        nlist = as.integer(nlist),
        nprobe = as.integer(nprobe),
        M = as.integer(m),
        n_bits = as.integer(n_bits),
        usePrecomputedTables = FALSE
      )
    ),
    class = "cuda_ml_knn_algo"
  )
}

#' Build a KNN model.
#'
#' Build a k-nearest-neighbor model for classification or regression tasks.
#'
#' @template supervised-model-inputs
#' @template supervised-model-output
#' @template ellipsis-unused
#' @param algo The query algorithm to use. For most workflows, pass one of
#'   \{"brute", "ivfflat", "ivfpq"\} or a KNN algorithm specification
#'   constructed using the \code{cuda_ml_knn_algo_*} family of functions.
#'   If the algorithm is specified by one of the \code{cuda_ml_knn_algo_*}
#'   functions, then values of all required parameters of the algorithm will
#'   need to be specified explicitly.
#'   If the algorithm is specified by a character vector, then parameters for
#'   the algorithm are generated automatically.
#'
#'   Descriptions of supported algorithms:
#'   - "brute": for brute-force, slow but produces exact results.
#'   - "ivfflat": for inverted file, divide the dataset in partitions and
#'     perform search on relevant partitions only.
#'   - "ivfpq": for inverted file and product quantization (vectors are
#'     divided into sub-vectors, and each sub-vector is encoded using
#'     intermediary k-means clusterings to provide partial information).
#'
#'   Default: "brute".
#' @param metric Distance metric to use. Must be one of \{"euclidean", "l2",
#'   "l1", "cityblock", "taxicab", "manhattan", "braycurtis", "canberra",
#'   "minkowski", "lp", "chebyshev", "linf", "jensenshannon", "cosine",
#'   "correlation"\}.
#'   The approximate algorithms support only "euclidean", "l2", "cosine", and
#'   "correlation".
#'   Default: "euclidean".
#' @param p Parameter for the Minkowski metric. If p = 1, then the metric is
#'   equivalent to manhattan distance (l1). If p = 2, the metric is equivalent
#'   to euclidean distance (l2).
#' @param neighbors Number of nearest neighbors to query. Default: 5L.
#'
#' @return A KNN model that can be used with the 'predict' S3 generic to make
#'   predictions on new data points.
#'   The model object contains the following:
#'   - "knn_index": a GPU pointer to the KNN index.
#'   - "algo": enum value of the algorithm being used for the KNN query.
#'   - "metric": enum value of the distance metric used in KNN computations.
#'   - "p": parameter for the Minkowski metric.
#'   - "n_samples": number of input data points.
#'   - "n_dims": dimension of each input data point.
#'
#' @examples
#'
#' library(cuda.ml)
#'
#' if (interactive() && cuda_ml_backend_info()$runtime_installed) {
#'   library(MASS)
#'   library(purrr)
#'
#'   set.seed(0L)
#'
#'   centers <- list(c(3, 3), c(-3, -3), c(-3, 3))
#'
#'   gen_pts <- function(cluster_sz) {
#'     pts <- centers |>
#'       map(\(center) mvrnorm(cluster_sz, mu = center, Sigma = diag(2)))
#'
#'     rlang::exec(rbind, !!!pts) |> as.matrix()
#'   }
#'
#'   gen_labels <- function(cluster_sz) {
#'     seq_along(centers) |>
#'       sapply(\(x) rep(x, cluster_sz)) |>
#'       factor()
#'   }
#'
#'   sample_cluster_sz <- 1000
#'   sample_pts <- cbind(
#'     gen_pts(sample_cluster_sz) |> as.data.frame(),
#'     label = gen_labels(sample_cluster_sz)
#'   )
#'
#'   model <- cuda_ml_knn(
#'     label ~ ., sample_pts, algo = "ivfflat", metric = "euclidean"
#'   )
#'
#'   test_cluster_sz <- 10
#'   test_pts <- gen_pts(test_cluster_sz) |> as.data.frame()
#'
#'   predictions <- predict(model, test_pts)
#'   print(predictions, n = 30)
#' }
#' @importFrom ellipsis check_dots_used
#' @export
cuda_ml_knn <- function(x, ...) {
  UseMethod("cuda_ml_knn")
}

#' @rdname cuda_ml_knn
#' @export
cuda_ml_knn.default <- function(x, ...) {
  report_undefined_fn("cuda_ml_knn", x)
}

#' @rdname cuda_ml_knn
#' @export
cuda_ml_knn.data.frame <- function(
  x,
  y,
  algo = c("brute", "ivfflat", "ivfpq"),
  metric = c(
    "euclidean",
    "l2",
    "l1",
    "cityblock",
    "taxicab",
    "manhattan",
    "braycurtis",
    "canberra",
    "minkowski",
    "lp",
    "chebyshev",
    "linf",
    "jensenshannon",
    "cosine",
    "correlation"
  ),
  p = 2.0,
  neighbors = 5L,
  ...
) {
  check_dots_used()
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
cuda_ml_knn.matrix <- function(
  x,
  y,
  algo = c("brute", "ivfflat", "ivfpq"),
  metric = c(
    "euclidean",
    "l2",
    "l1",
    "cityblock",
    "taxicab",
    "manhattan",
    "braycurtis",
    "canberra",
    "minkowski",
    "lp",
    "chebyshev",
    "linf",
    "jensenshannon",
    "cosine",
    "correlation"
  ),
  p = 2.0,
  neighbors = 5L,
  ...
) {
  check_dots_used()
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
cuda_ml_knn.formula <- function(
  formula,
  data,
  algo = c("brute", "ivfflat", "ivfpq"),
  metric = c(
    "euclidean",
    "l2",
    "l1",
    "cityblock",
    "taxicab",
    "manhattan",
    "braycurtis",
    "canberra",
    "minkowski",
    "lp",
    "chebyshev",
    "linf",
    "jensenshannon",
    "cosine",
    "correlation"
  ),
  p = 2.0,
  neighbors = 5L,
  ...
) {
  check_dots_used()
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
cuda_ml_knn.recipe <- function(
  x,
  data,
  algo = c("brute", "ivfflat", "ivfpq"),
  metric = c(
    "euclidean",
    "l2",
    "l1",
    "cityblock",
    "taxicab",
    "manhattan",
    "braycurtis",
    "canberra",
    "minkowski",
    "lp",
    "chebyshev",
    "linf",
    "jensenshannon",
    "cosine",
    "correlation"
  ),
  p = 2.0,
  neighbors = 5L,
  ...
) {
  check_dots_used()
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

  if (is.factor(y)) {
    validate_classification_outcome(y)
  }
  stopifnot(
    "`p` must be one positive finite number" = is.numeric(p) &&
      length(p) == 1L &&
      is.finite(p) &&
      p > 0,
    "`neighbors` must be one positive whole number no greater than the number of training rows" = is.numeric(
      neighbors
    ) &&
      length(neighbors) == 1L &&
      is.finite(neighbors) &&
      neighbors >= 1L &&
      neighbors == as.integer(neighbors) &&
      neighbors <= nrow(x)
  )

  if (is.character(algo)) {
    algo_type <- knn_match_algo(algo)
    algo_params <- list()
  } else {
    stopifnot(
      "`algo` must be a KNN algorithm specification" = inherits(
        algo,
        "cuda_ml_knn_algo"
      ) &&
        is.list(algo) &&
        identical(names(algo), c("type", "params")) &&
        is.integer(algo$type) &&
        length(algo$type) == 1L &&
        algo$type %in% c(1L, 2L) &&
        is.list(algo$params)
    )
    algo_type <- algo$type
    algo_params <- algo$params
    expected_params <- if (algo_type == 1L) {
      c("nlist", "nprobe")
    } else {
      c("nlist", "nprobe", "M", "n_bits", "usePrecomputedTables")
    }
    stopifnot(
      "The KNN algorithm specification has invalid parameters" = identical(
        names(algo_params),
        expected_params
      ),
      "`nlist` must not exceed the number of training rows" = algo_params$nlist <=
        nrow(x),
      "`m` must divide the number of predictors for IVFPQ" = algo_type != 2L ||
        ncol(x) %% algo_params$M == 0L
    )
  }
  metric <- knn_match_metric(metric)
  stopifnot(
    "Approximate KNN algorithms support only `euclidean`, `l2`, `cosine`, and `correlation` metrics" = algo_type ==
      0L ||
      metric %in% c(1L, 2L, 10L)
  )

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
#' Make predictions on new data points using a cuML KNN model.
#'
#' @template predict
#' @param type Type of prediction. Classification models support
#'   \code{"class"} and \code{"prob"}; regression models support
#'   \code{"numeric"}. The default is \code{"class"} for classification and
#'   \code{"numeric"} for regression.
#'
#' @importFrom ellipsis check_dots_used
#' @export
predict.cuda_ml_knn <- function(object, new_data, type = NULL, ...) {
  check_dots_used()

  processed <- hardhat::forge(new_data, object$blueprint)

  predict_cuda_ml_knn_bridge(
    model = object,
    processed = processed,
    type = type
  )
}

predict_cuda_ml_knn_bridge <- function(model, processed, type) {
  out <- switch(
    model$mode,
    classification = {
      type <- match.arg(type %||% "class", c("class", "prob"))
      predict_cuda_ml_knn_classification_impl(
        model = model,
        processed = processed,
        type = type
      )
    },
    regression = {
      type <- match.arg(type %||% "numeric", "numeric")

      predict_cuda_ml_knn_regression_impl(
        model = model,
        processed = processed
      )
    }
  )
  hardhat::validate_prediction_size(out, processed$predictors)

  out
}

predict_cuda_ml_knn_classification_impl <- function(model, processed, type) {
  if (identical(type, "prob")) {
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
      model = "nearest_neighbor",
      mode = mode,
      eng = pkgname
    )
  }

  parsnip::set_dependency(
    model = "nearest_neighbor",
    eng = pkgname,
    pkg = pkgname
  )

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
        allow_sparse_x = FALSE
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
          object = quote(object$fit),
          new_data = quote(new_data),
          type = type
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
