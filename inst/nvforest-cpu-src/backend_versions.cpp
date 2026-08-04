#include <nvforest/version_config.hpp>
#include <treelite/version.h>

#include <Rcpp.h>

static_assert(NVForest_VERSION_MAJOR == 26 && NVForest_VERSION_MINOR == 6 &&
              NVForest_VERSION_PATCH == 0);
static_assert(TREELITE_VER_MAJOR == 4 && TREELITE_VER_MINOR == 7 &&
              TREELITE_VER_PATCH == 0);

Rcpp::List nvforest_backend_versions() {
  return Rcpp::List::create(Rcpp::Named("nvforest") = "26.06.0",
                            Rcpp::Named("treelite") = "4.7.0");
}
