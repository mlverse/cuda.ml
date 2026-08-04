# Package index

## Install and manage the backend

- [`cuda_ml_install()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_install.md)
  : Install a cuda.ml native backend
- [`cuda_ml_backend_info()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_backend_info.md)
  : Report native-backend metadata
- [`cuda_ml_runtime_audit()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_runtime_audit.md)
  : Audit the installed native backend
- [`cuda_ml_cache_clean()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_cache_clean.md)
  : Remove cuda.ml native-backend caches

## Linear models

- [`cuda_ml_linear_reg()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_linear_reg.md)
  : Train a regularized linear regression model
- [`cuda_ml_logistic_reg()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_logistic_reg.md)
  : Train a logistic or multinomial regression model
- [`cuda_ml_ols()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_ols.md)
  : Train an OLS model.
- [`cuda_ml_ridge()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_ridge.md)
  : Train a linear model using ridge regression.
- [`cuda_ml_lasso()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_lasso.md)
  : Train a linear model using LASSO regression.
- [`cuda_ml_elastic_net()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_elastic_net.md)
  : Train a linear model using elastic net regression.
- [`cuda_ml_sgd()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_sgd.md)
  : Train a linear model using mini-batch stochastic gradient descent.
- [`predict(`*`<cuda_ml_linear_model>`*`)`](https://mlverse.github.io/cuda.ml/reference/predict.cuda_ml_linear_model.md)
  : Make predictions on new data points.
- [`predict(`*`<cuda_ml_logistic_reg>`*`)`](https://mlverse.github.io/cuda.ml/reference/predict.cuda_ml_logistic_reg.md)
  : Predict from a logistic or multinomial regression model

## Nonlinear models

- [`cuda_ml_knn()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_knn.md)
  : Build a KNN model.
- [`cuda_ml_knn_algo_ivfflat()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_knn_algo_ivfflat.md)
  : Build a specification for the "ivfflat" KNN query algorithm.
- [`cuda_ml_knn_algo_ivfpq()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_knn_algo_ivfpq.md)
  : Build a specification for the "ivfpq" KNN query algorithm.
- [`cuda_ml_rand_forest()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_rand_forest.md)
  : Train a random forest model
- [`cuda_ml_svm()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_svm.md)
  : Train a SVM model.
- [`predict(`*`<cuda_ml_knn>`*`)`](https://mlverse.github.io/cuda.ml/reference/predict.cuda_ml_knn.md)
  : Make predictions on new data points.
- [`predict(`*`<cuda_ml_svm>`*`)`](https://mlverse.github.io/cuda.ml/reference/predict.cuda_ml_svm.md)
  : Make predictions on new data points.

## Clustering

- [`cuda_ml_agglomerative_clustering()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_agglomerative_clustering.md)
  : Perform single-linkage agglomerative clustering.
- [`cuda_ml_dbscan()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_dbscan.md)
  : Run the DBSCAN clustering algorithm.
- [`cuda_ml_kmeans()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_kmeans.md)
  : Run the k-means clustering algorithm.

## Dimensionality reduction

- [`cuda_ml_pca()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_pca.md)
  : Perform principal component analysis.
- [`cuda_ml_tsvd()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_tsvd.md)
  : Truncated SVD.
- [`cuda_ml_umap()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_umap.md)
  : Uniform Manifold Approximation and Projection (UMAP) for dimension
  reduction.
- [`cuda_ml_tsne()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_tsne.md)
  : Perform t-distributed stochastic neighbor embedding.
- [`cuda_ml_transform()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_transform.md)
  : Transform data using a trained cuML model.
- [`cuda_ml_inverse_transform()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_inverse_transform.md)
  : Apply the inverse transformation defined by a trained cuML model.

## nvForest inference

- [`cuda_ml_nvforest_load_model()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_load_model.md)
  : Load a tree ensemble with nvForest
- [`predict(`*`<cuda_ml_nvforest>`*`)`](https://mlverse.github.io/cuda.ml/reference/predict.cuda_ml_nvforest.md)
  : Predict with an nvForest model
- [`cuda_ml_nvforest_info()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_info.md)
  : Inspect an nvForest model
- [`cuda_ml_nvforest_leaf_ids()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_leaf_ids.md)
  : Return terminal leaf identifiers
- [`cuda_ml_nvforest_predict_per_tree()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_predict_per_tree.md)
  : Return individual-tree predictions

## Model persistence

- [`cuda_ml_serialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_serialize.md)
  : Serialize a cuML model
- [`cuda_ml_unserialize()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_unserialize.md)
  : Unserialize a cuML model state
- [`bundle(`*`<cuda_ml_model>`*`)`](https://mlverse.github.io/cuda.ml/reference/bundle.cuda_ml_model.md)
  [`bundle(`*`<cuda_ml_nvforest>`*`)`](https://mlverse.github.io/cuda.ml/reference/bundle.cuda_ml_model.md)
  : Bundle a cuda.ml model
- [`cuda_ml_nvforest_export()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md)
  [`cuda_ml_nvforest_import()`](https://mlverse.github.io/cuda.ml/reference/cuda_ml_nvforest_export.md)
  : Export and import an nvForest checkpoint pair

## Package overview

- [`cuda.ml`](https://mlverse.github.io/cuda.ml/reference/cuda.ml-package.md)
  [`cuda.ml-package`](https://mlverse.github.io/cuda.ml/reference/cuda.ml-package.md)
  : cuda.ml
