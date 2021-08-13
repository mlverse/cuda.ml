#include <treelite/c_api.h>

// [[Rcpp::export(".fil_enabled")]]
bool fil_enabled() {
#ifdef CUML4R_TREELITE_C_API_MISSING
  return false;
#else
  return true;
#endif
}
