test_that("only current KNN algorithms are exposed", {
  exports <- getNamespaceExports("cuda.ml")

  expect_true("cuda_ml_knn_algo_ivfflat" %in% exports)
  expect_true("cuda_ml_knn_algo_ivfpq" %in% exports)
  expect_false("cuda_ml_knn_algo_ivfsq" %in% exports)
})

test_that("IVFPQ exposes only supported parameters", {
  expect_identical(
    names(formals(cuda_ml_knn_algo_ivfpq)),
    c("nlist", "nprobe", "m", "n_bits")
  )

  specification <- cuda_ml_knn_algo_ivfpq(4, 2, 2, 4)
  expect_identical(specification$params$usePrecomputedTables, FALSE)
})

test_that("obsolete random projection and FIL interfaces are absent", {
  exports <- getNamespaceExports("cuda.ml")

  expect_false("cuda_ml_rand_proj" %in% exports)
  expect_false("cuda_ml_rand_proj_enabled" %in% exports)
  expect_false("cuda_ml_fil_load_model" %in% exports)
  expect_false("cuda_ml_fil_enabled" %in% exports)
})

test_that("prediction methods use the R modeling new_data convention", {
  methods <- c(
    "predict.cuda_ml_knn",
    "predict.cuda_ml_linear_model",
    "predict.cuda_ml_logistic_reg",
    "predict.cuda_ml_nvforest",
    "predict.cuda_ml_svm"
  )

  for (method in methods) {
    expect_true("new_data" %in% names(formals(get(method))))
  }
})

test_that("classification rejects unobserved outcome levels", {
  data <- penguins[penguins$species != "Gentoo", ]

  expect_error(
    cuda_ml_rand_forest(species ~ ., data, trees = 10L),
    "Every outcome factor level must be represented"
  )
  expect_error(
    cuda_ml_logistic_reg(species ~ ., data),
    "Every outcome factor level must be represented"
  )
  expect_error(
    cuda_ml_knn(species ~ ., data, algo = "brute", metric = "euclidean"),
    "Every outcome factor level must be represented"
  )
  expect_error(
    cuda_ml_svm(species ~ ., data, cost = 1),
    "Every outcome factor level must be represented"
  )
})

test_that("native operations fail clearly on unsupported platforms", {
  skip_if(native_platform_supported, "requires an unsupported platform")

  expect_error(
    cuda_ml_knn(mpg ~ ., mtcars),
    "requires Linux x86_64 with glibc 2.28 or newer",
    fixed = TRUE
  )
})

test_that("KNN defaults reach backend selection without a runtime", {
  skip_if_not(
    native_platform_supported,
    "requires the native backend platform"
  )
  info <- cuda_ml_backend_info()
  skip_if(info$runtime_installed)
  error <- if (info$backend_available) {
    "cuda_ml_install"
  } else {
    "No prebuilt cuda.ml backend"
  }

  expect_error(
    cuda_ml_knn(mpg ~ ., mtcars),
    error
  )
})

test_that("KNN parameters fail before backend execution", {
  expect_error(cuda_ml_knn_algo_ivfflat(0, 1), "nlist")
  expect_error(cuda_ml_knn_algo_ivfflat(4, 5), "nprobe")
  expect_error(
    cuda_ml_knn_algo_ivfpq(4, 2, 2, 0),
    "n_bits"
  )
  expect_error(
    cuda_ml_knn_algo_ivfpq(4, 2, 3, 4),
    "divisible by 8"
  )
  expect_error(
    cuda_ml_knn_algo_ivfpq(4, 2, 2, 9),
    "n_bits"
  )

  specification <- cuda_ml_knn_algo_ivfpq(4, 2, 3, 8)
  expect_s3_class(specification, "cuda_ml_knn_algo")
  expect_error(
    cuda_ml_knn(mpg ~ ., mtcars, algo = specification),
    "m.*divide"
  )
  expect_error(cuda_ml_knn(mpg ~ ., mtcars, p = 0), "p")
  expect_error(
    cuda_ml_knn(mpg ~ ., mtcars, neighbors = nrow(mtcars) + 1L),
    "neighbors"
  )
  expect_error(
    cuda_ml_knn(mpg ~ ., mtcars, algo = list(type = 1L, params = list())),
    "algorithm specification"
  )
})

test_that("approximate KNN rejects unsupported metrics", {
  unsupported <- c(
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
    "jensenshannon"
  )

  for (algo in c("ivfflat", "ivfpq")) {
    for (metric in unsupported) {
      expect_error(
        cuda_ml_knn(mpg ~ ., mtcars, algo = algo, metric = metric),
        "Approximate KNN algorithms support only",
        label = paste(algo, metric)
      )
    }
  }
})

test_that("default fit calls request the RAPIDS off log level", {
  verbosity <- new.env(parent = emptyenv())
  local_mocked_bindings(
    .dbscan = function(...) {
      args <- list(...)
      verbosity$dbscan <- args$verbosity
      list(labels = integer(nrow(args$x)))
    },
    .kmeans = function(...) {
      args <- list(...)
      verbosity$kmeans <- args$verbosity
      list()
    },
    .tsne_fit = function(...) {
      args <- list(...)
      verbosity$tsne <- args$verbosity
      matrix(0, nrow(args$x), args$dim)
    },
    .umap_fit = function(...) {
      args <- list(...)
      verbosity$umap <- args$verbosity
      list()
    },
    .svc_fit = function(...) {
      args <- list(...)
      verbosity$svm <- args$verbosity
      NULL
    },
    .package = "cuda.ml"
  )

  cuda_ml_dbscan(matrix(c(0, 1), ncol = 1), min_pts = 1, eps = 1)
  cuda_ml_kmeans(matrix(c(0, 1), ncol = 1), k = 1)
  cuda_ml_tsne(matrix(seq_len(12), nrow = 6), method = "exact")
  cuda_ml_umap(matrix(seq_len(12), nrow = 6), transform_input = FALSE)
  data <- penguins[penguins$species != "Gentoo", ]
  data$species <- droplevels(data$species)
  cuda_ml_svm(species ~ ., data)

  expect_identical(verbosity$dbscan, 6L)
  expect_identical(verbosity$kmeans, 6L)
  expect_identical(verbosity$tsne, 6L)
  expect_identical(verbosity$umap, 6L)
  expect_identical(verbosity$svm, 6L)
})

test_that("random forest mtry survives conversion to single precision", {
  forwarded <- new.env(parent = emptyenv())
  local_mocked_bindings(
    .rf_classifier_fit = function(...) {
      args <- list(...)
      forwarded$max_features <- args$max_features
      NULL
    },
    .nvforest_model_info = function(...) {
      list(task_type = 2L, num_classes = 2L)
    },
    .package = "cuda.ml"
  )
  data <- as.data.frame(matrix(seq_len(10L * 37L), nrow = 10L))
  data$outcome <- factor(rep(c("a", "b"), 5L))

  cuda_ml_rand_forest(outcome ~ ., data, trees = 1L, seed = 0L)

  raw_float <- writeBin(forwarded$max_features, raw(), size = 4L)
  max_features <- readBin(raw_float, double(), n = 1L, size = 4L)
  expect_identical(as.integer(max_features * 37L), 6L)
})

test_that("random forest arguments fail before native execution", {
  expect_error(
    cuda_ml_rand_forest(species ~ ., penguins, split_criterion = "mse"),
    "one of"
  )
  expect_error(
    cuda_ml_rand_forest(mpg ~ ., mtcars, split_criterion = "gini"),
    "one of"
  )
  expect_error(
    cuda_ml_rand_forest(mpg ~ ., mtcars, split_criterion = "mae"),
    "one of"
  )
  expect_error(
    cuda_ml_rand_forest(mpg ~ ., mtcars, n_streams = 0L),
    "n_streams"
  )
  expect_error(
    cuda_ml_rand_forest(mpg ~ ., mtcars, seed = NA_integer_),
    "seed"
  )
})

test_that("logistic regression arguments fail before native execution", {
  data <- penguins[penguins$species != "Gentoo", ]
  data$species <- droplevels(data$species)

  expect_error(
    cuda_ml_logistic_reg(species ~ ., data, fit_intercept = NA),
    "fit_intercept"
  )
  expect_error(
    cuda_ml_logistic_reg(species ~ ., data, tol = 0),
    "tol"
  )
  expect_error(
    cuda_ml_logistic_reg(species ~ ., data, max_iter = 1.5),
    "max_iter"
  )
})
