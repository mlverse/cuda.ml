#include "preprocessor.h"

#ifdef HAS_CUML

#include <cuml/version_config.hpp>

static_assert(CUML_VERSION_MAJOR == 21,
              "{cuda.ml} currently only supports linking to RAPIDS cuML 21.x!");

#endif

#include <Rcpp.h>

// [[Rcpp::export(".has_cuML")]]
bool has_cuML() {
#ifdef HAS_CUML

  return true;

#else

  return false;

#endif
}

// [[Rcpp::export(".cuML_major_version")]]
Rcpp::CharacterVector cuML_major_version() {
#ifdef HAS_CUML

  return CUML4R_TO_STRING(CUML_VERSION_MAJOR);

#else

  return NA_STRING;

#endif
}

// [[Rcpp::export(".cuML_minor_version")]]
Rcpp::CharacterVector cuML_minor_version() {
#ifdef HAS_CUML

  return CUML4R_TO_STRING(CUML_VERSION_MINOR);

#else

  return NA_STRING;

#endif
}
