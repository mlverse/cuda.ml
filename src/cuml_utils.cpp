#ifdef HAS_CUML

#include <cuml/version_config.hpp>

static_assert(
  CUML_VERSION_MAJOR == 21,
  "{cuda.ml} currently only supports linking to RAPIDS cuML 21.x!"
);

#endif

// [[Rcpp::export(".has_libcuml")]]
bool has_libcuml() {
#ifdef HAS_CUML

  return true;

#else

  return false;

#endif
}
