#pragma once

#if HAS_CUML

#include <cuml/ensemble/randomforest.hpp>

#include <memory>
#include <unordered_map>

namespace cuml4r {

template <typename T, typename L>
struct RandomForestMetaDataDeleter {
  void operator()(ML::RandomForestMetaData<T, L>* const rf) const {
    ML::delete_rf_metadata<T, L>(rf);
  }
};

using RandomForestClassifierUPtr =
  std::unique_ptr<ML::RandomForestClassifierD,
                  RandomForestMetaDataDeleter<double, int>>;

using RandomForestRegressorUPtr =
  std::unique_ptr<ML::RandomForestRegressorD,
                  RandomForestMetaDataDeleter<double, double>>;

struct RandomForestClassifierModel {
  RandomForestClassifierUPtr const rf_;
  std::unordered_map<int, int> const inverseLabelsMap_;
  RandomForestClassifierModel(
    RandomForestClassifierUPtr rf,
    std::unordered_map<int, int>&& inverse_labels_map) noexcept
    : rf_(std::move(rf)), inverseLabelsMap_(std::move(inverse_labels_map)) {}
};

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
