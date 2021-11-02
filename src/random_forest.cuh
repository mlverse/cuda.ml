#pragma once

#include "preprocessor.h"
#include "treelite_utils.cuh"

#include <cuml/ensemble/randomforest.hpp>
#include <cuml/version_config.hpp>

namespace cuml4r {

#ifndef CUML4R_TREELITE_C_API_MISSING

namespace detail {

template <typename T, typename L>
__host__ TreeliteHandle
build_treelite_forest(ML::RandomForestMetaData<T, L> const* forest,
                      int const n_features, int const n_classes) {
  TreeliteHandle handle;

#if (CUML4R_LIBCUML_VERSION(CUML_VERSION_MAJOR, CUML_VERSION_MINOR) >= \
     CUML4R_LIBCUML_VERSION(21, 10))

  ML::build_treelite_forest(/*model=*/handle.get(), forest,
                            /*num_features=*/n_features);

#else

  ML::build_treelite_forest(/*model=*/handle.get(), forest,
                            /*num_features=*/n_features,
                            /*task_category=*/n_classes);

#endif

  return handle;
}

}  // namespace detail

#endif

template <typename T, typename L>
struct RandomForestMetaDataDeleter {
  __host__ void operator()(ML::RandomForestMetaData<T, L>* const rf) const {
    ML::delete_rf_metadata<T, L>(rf);
  }
};

}  // namespace cuml4r
