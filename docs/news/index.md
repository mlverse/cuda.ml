# Changelog

## cuda.ml 0.4.0

CRAN release: 2026-08-21

This is a breaking release for users upgrading from the cuda.ml 0.3
release series. It requires R 4.1 or newer and targets CUDA Toolkit
13.2.2, RAPIDS cuML and nvForest 26.06, and Treelite 4.7.0.

### Installation and runtime

- The CRAN package is now a portable R installer and loader with no
  compiled code or bundled CUDA libraries. On Linux x86_64 with glibc
  2.28 or newer,
  [`cuda_ml_install()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_install.md)
  explicitly prepares the pinned native backend and its approximately
  1.6 GiB CUDA and RAPIDS runtime. Repeated calls reuse the prepared
  cache.

- `cuda_ml_install(source = TRUE)` builds the backend on a supported
  host. Its managed mode prepares the pinned CUDA, RAPIDS, Treelite,
  CMake, and Ninja build inputs and requires GNU C++ 14 or newer. Use
  `dependencies = "host"` to supply all native build inputs. Managed
  builds detect distinct CUDA-visible architectures when possible, while
  `architectures = "native"`, `"portable"`, or an explicit CMake
  architecture list selects the build target policy.

- Loading cuda.ml is silent; it does not inspect a GPU, load native
  code, modify the cache, or use the network.
  [`cuda_ml_backend_info()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_backend_info.md)
  reports pinned versions and cache status without those actions.
  [`cuda_ml_runtime_audit()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_runtime_audit.md)
  is the separate, explicit full content and native registration check,
  and
  [`cuda_ml_cache_clean()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_cache_clean.md)
  removes managed cache generations.

- CPU nvForest inference requires a prepared cuda.ml backend. It can use
  the complete runtime installed by
  [`cuda_ml_install()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_install.md)
  or the separate CPU-only backend installed by
  `cuda_ml_install(device = "cpu")`. Selecting `device = "cpu"` when
  loading or restoring a model does not itself prepare a backend.

- Prebuilt GPU backends contain real targets for compute capabilities
  7.5, 8.0, 8.6, 8.9, 9.0, 10.0, and 12.0, plus PTX forward
  compatibility from 12.0.

### Models and public APIs

- The former FIL interface was replaced by nvForest. Use
  [`cuda_ml_nvforest_load_model()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_load_model.md)
  for XGBoost models, LightGBM text models, and Treelite checkpoints,
  standard [`predict()`](https://rdrr.io/r/stats/predict.html) for
  inference, and the `cuda_ml_nvforest_*()` inspection, checkpoint
  export, and import functions for nvForest-specific operations.
  Random-forest fits now use nvForest for prediction and persistence as
  well.

- The random-forest API changed. `mtry` controls predictor sampling,
  `sample_fraction` controls row sampling and defaults to 1, and the
  separate `max_predictors_per_note_split` argument was removed. When
  `mtry` is omitted, classification uses the square root of the
  predictor count and regression uses all predictors. `trees` defaults
  to 100. When `seed` is omitted, it is drawn from R’s random-number
  generator, so [`set.seed()`](https://rdrr.io/r/base/Random.html)
  controls the fit. The `max_batch_size` and `n_streams` defaults are
  now 4096 and 4. Regression split criteria are now `"mse"`,
  `"poisson"`, `"gamma"`, and `"inverse_gaussian"`; `"mae"` was removed.
  Classification predictions use `predict(..., type = "class")` or
  `predict(..., type = "prob")`.

- Logistic and multinomial regression now use numeric `penalty` and
  `mixture` arguments consistent with parsnip instead of the previous
  `penalty`, `C`, and `l1_ratio` combination. The default is now
  unregularized; set a numeric `penalty` to request regularization. The
  iteration arguments are now `max_iter` and `linesearch_max_iter`;
  `lbfgs_memory` and `penalty_normalized` provide additional solver
  controls.
  [`cuda_ml_linear_reg()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_linear_reg.md)
  provides the corresponding parsnip-style routing across OLS, ridge,
  lasso, and elastic-net fits.

- The `normalize_input` argument was removed from OLS, ridge, lasso, and
  elastic-net models. It previously requested GPU-side L2 normalization.
  [`recipes::step_normalize()`](https://recipes.tidymodels.org/reference/step_normalize.html)
  is the recommended explicit preprocessing step when centering and
  scaling are appropriate, but it is not numerically identical to the
  former L2 operation.

- [`cuda_ml_sgd()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_sgd.md)
  now fits squared-loss regression only, so its `loss` argument was
  removed, and `n_iters_no_change` was renamed to `n_iter_no_change`.
  Prediction methods now consistently use `new_data`; KNN classification
  uses `type = "class"` or `type = "prob"` instead of
  `output_class_probabilities`.

- Parsnip is optional. cuda.ml registers engines when parsnip is loaded,
  including
  [`linear_reg()`](https://parsnip.tidymodels.org/reference/linear_reg.html),
  [`logistic_reg()`](https://parsnip.tidymodels.org/reference/logistic_reg.html),
  and
  [`multinom_reg()`](https://parsnip.tidymodels.org/reference/multinom_reg.html)
  engines that use the usual `penalty` and `mixture` arguments.

- Random projection and KNN IVFSQ were removed and have no current
  replacement in the pinned upstream API. KNN continues to support
  brute-force, IVFFlat, and IVFPQ search. The unused
  `use_precomputed_tables` argument was removed from
  [`cuda_ml_knn_algo_ivfpq()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_knn_algo.md).

- The per-call `cuML_log_level` arguments were removed. `has_cuML()`,
  `cuML_major_version()`, and `cuML_minor_version()` were removed in
  favor of fields returned by
  [`cuda_ml_backend_info()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_backend_info.md).

- `cuda_ml_is_classifier()` and
  `cuda_ml_can_predict_class_probabilities()` were removed. Use the
  documented [`predict()`](https://rdrr.io/r/stats/predict.html) types
  for each model; for nvForest models,
  [`cuda_ml_nvforest_info()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_info.md)
  reports `task_type` and `has_probability_output`. The
  `cuda_ml_serialise()` and `cuda_ml_unserialise()` aliases were also
  removed; use
  [`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
  and
  [`cuda_ml_unserialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md).

### Model persistence

- [`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
  and
  [`cuda_ml_unserialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
  now provide durable model states for OLS, ridge, lasso, elastic-net,
  SGD, logistic and multinomial regression, PCA, binary and one-vs-rest
  SVC, SVR, UMAP, random forest, and nvForest models. Passing a file
  path writes or reads a gzip-compressed state; open connections and
  in-memory raw vectors are also supported. KNN and TSVD fits are not
  currently supported: the pinned KNN API does not expose portable
  approximate-index state, and the current TSVD binding does not
  reconstruct its native transform parameters.

- [`bundle::bundle()`](https://rstudio.github.io/bundle/reference/bundle.html)
  stores the same explicit state for workflows that use the bundle
  package. Both interfaces support saving an artifact and restoring it
  in a fresh R process after the target environment prepares the
  required backend. nvForest models can also be exported and imported as
  a Treelite checkpoint plus cuda.ml metadata.

- cuda.ml validates each saved state and its required backend before
  restoring the model.

### Documentation

- The function reference is reorganized and includes guides for getting
  started, installation and runtime management, tidymodels, model
  persistence, and nvForest inference and deployment.
