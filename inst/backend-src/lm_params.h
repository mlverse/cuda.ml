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
  // output on the host
  double* intercept;
  // LM settings
  bool fit_intercept;
};

}  // namespace lm
}  // namespace cuml4r
