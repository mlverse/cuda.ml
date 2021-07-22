#pragma once

#if HAS_CUML

#include <cuml/ensemble/randomforest.hpp>

namespace cuml4r {

template <typename T, typename L>
struct RandomForestMetaDataDeleter {
  void operator()(ML::RandomForestMetaData<T, L>* const rf) const {
    ML::delete_rf_metadata<T, L>(rf);
  }
};

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
