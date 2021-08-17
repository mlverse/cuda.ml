#pragma once

#ifdef HAS_CUML

#include <cuml/ensemble/randomforest.hpp>

namespace cuml4r {

template <typename T, typename L>
struct RandomForestMetaDataDeleter {
  __host__ void operator()(ML::RandomForestMetaData<T, L>* const rf) const {
    ML::delete_rf_metadata<T, L>(rf);
  }
};

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
