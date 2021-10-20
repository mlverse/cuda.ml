#pragma once

namespace cuml4r {
namespace lm {

struct Params {
  // LM input
  double* d_input;
  int n_rows;
  int n_cols;
  double* d_labels;
  // LM output
  double* d_coef;
  double* d_intercept;
  // LM settings
  bool fit_intercept;
  bool normalize_input;
};

}  // namespace lm
}  // namespace cuml4r
