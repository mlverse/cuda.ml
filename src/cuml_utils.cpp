#include "preprocessor.h"

#include <cuda_runtime_api.h>
#include <treelite/version.h>
#include <cuml/version_config.hpp>
#include <nvforest/version_config.hpp>

static_assert(CUML_VERSION_MAJOR == 26 && CUML_VERSION_MINOR == 6 &&
                CUML_VERSION_PATCH == 0,
              "{cuda.ml} requires RAPIDS cuML 26.06.");
static_assert(NVForest_VERSION_MAJOR == 26 && NVForest_VERSION_MINOR == 6 &&
                NVForest_VERSION_PATCH == 0,
              "{cuda.ml} requires nvForest 26.06.0.");
static_assert(TREELITE_VER_MAJOR == 4 && TREELITE_VER_MINOR == 6 &&
                TREELITE_VER_PATCH == 1,
              "{cuda.ml} requires Treelite 4.6.1.");

#include <Rcpp.h>

// [[Rcpp::export(".backend_versions")]]
Rcpp::List backend_versions() {
  return Rcpp::List::create(Rcpp::Named("cuml") = "26.06",
                            Rcpp::Named("nvforest") = "26.06.0",
                            Rcpp::Named("treelite") = "4.6.1",
                            Rcpp::Named("cuda_runtime") = CUDART_VERSION);
}
