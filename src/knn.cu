#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "random_forest.cuh"
#include "stream_allocator.h"

#include <raft/spatial/knn/ann_common.h>
#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/neighbors/knn.hpp>

#include <Rcpp.h>

#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

char const* const N_LIST = "nlist";
char const* const N_PROBE = "nprobe";
char const* const M_VALUE = "M";
char const* const N_BITS = "n_bits";
char const* const USE_PRE_COMPUTED_TABLES = "usePrecomputedTables";
char const* const Q_TYPE = "qtype";
char const* const ENCODE_RESIDUAL = "encodeResidual";

std::unordered_map<std::string, raft::spatial::knn::QuantizerType> const
  kQuantizerTypes{
    {"QT_8bit", raft::spatial::knn::QuantizerType::QT_8bit},
    {"QT_4bit", raft::spatial::knn::QuantizerType::QT_4bit},
    {"QT_8bit_uniform", raft::spatial::knn::QuantizerType::QT_8bit_uniform},
    {"QT_4bit_uniform", raft::spatial::knn::QuantizerType::QT_4bit_uniform},
    {"QT_fp16", raft::spatial::knn::QuantizerType::QT_fp16},
    {"QT_8bit_direct", raft::spatial::knn::QuantizerType::QT_8bit_direct},
    {"QT_6bit", raft::spatial::knn::QuantizerType::QT_6bit}};

// Additional info for setting KNN params
struct ParamsDetails {
  int numRows_;
  int numCols_;
};

enum class KnnAlgo { BRUTE_FORCE = 0, IVFFLAT = 1, IVFPQ = 2, IVFSQ = 3 };

__host__ void validate_param_list(
  Rcpp::List const& params, std::initializer_list<char const*> required_attrs) {
  for (auto const attr : required_attrs) {
    if (!params.containsElementNamed(attr)) {
      Rcpp::stop("Required attribute '%s' is missing from algo params!", attr);
    }
  }
}

__host__ void validate_algo_params(KnnAlgo const algo,
                                   Rcpp::List const& params) {
  if (algo == KnnAlgo::IVFFLAT) {
    validate_param_list(params, {N_LIST, N_PROBE});
  } else if (algo == KnnAlgo::IVFPQ) {
    validate_param_list(
      params, {N_LIST, N_PROBE, M_VALUE, N_BITS, USE_PRE_COMPUTED_TABLES});
  } else if (algo == KnnAlgo::IVFSQ) {
    validate_param_list(params, {N_LIST, N_PROBE, Q_TYPE, ENCODE_RESIDUAL});
  }
}

__host__ std::unique_ptr<raft::spatial::knn::knnIndexParam>
build_ivfflat_algo_params(Rcpp::List params, bool const automated) {
  if (automated) {
    params[N_LIST] = 8;
    params[N_PROBE] = 2;
  }

  auto algo_params = std::make_unique<raft::spatial::knn::IVFFlatParam>();
  algo_params->nlist = params[N_LIST];
  algo_params->nprobe = params[N_PROBE];

  return algo_params;
}

__host__ std::unique_ptr<raft::spatial::knn::knnIndexParam>
build_ivfpq_algo_params(Rcpp::List params, bool const automated,
                        ParamsDetails const& details) {
  constexpr std::array<int, 13> kAllowedSubquantizers = {
    1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48};
  constexpr std::array<int, 13> kAllowedSubDimSize = {1,  2,  3,  4,  6,  8, 10,
                                                      12, 16, 20, 24, 28, 32};

  if (automated) {
    auto const n = details.numRows_;
    auto const d = details.numCols_;

    params[N_LIST] = 8;
    params[N_PROBE] = 3;

    for (auto const n_subq : kAllowedSubquantizers) {
      if (d % n_subq == 0 &&
          std::find(kAllowedSubDimSize.cbegin(), kAllowedSubDimSize.cend(),
                    d / n_subq) != kAllowedSubDimSize.cend()) {
        params[USE_PRE_COMPUTED_TABLES] = false;
        params[M_VALUE] = n_subq;
        break;
      }
    }

    if (!params.containsElementNamed(M_VALUE)) {
      for (auto const n_subq : kAllowedSubquantizers) {
        if (d % n_subq == 0) {
          params[USE_PRE_COMPUTED_TABLES] = true;
          params[M_VALUE] = n_subq;
          break;
        }
      }
    }

    params[N_BITS] = 4;
    for (auto const n_bits : {8, 6, 5}) {
      auto const min_train_points = (1 << n_bits) * 39;
      if (n >= min_train_points) {
        params[N_BITS] = n_bits;
        break;
      }
    }
  }

  auto algo_params = std::make_unique<raft::spatial::knn::IVFPQParam>();
  algo_params->nlist = Rcpp::as<int>(params[N_LIST]);
  algo_params->nprobe = Rcpp::as<int>(params[N_PROBE]);
  algo_params->M = Rcpp::as<int>(params[M_VALUE]);
  algo_params->n_bits = Rcpp::as<int>(params[N_BITS]);
  algo_params->usePrecomputedTables =
    Rcpp::as<bool>(params[USE_PRE_COMPUTED_TABLES]);

  return algo_params;
}

__host__ std::unique_ptr<raft::spatial::knn::knnIndexParam>
build_ivfsq_algo_params(Rcpp::List params, bool const automated) {
  if (automated) {
    params[N_LIST] = 8;
    params[N_PROBE] = 2;
    params[Q_TYPE] = "QT_8bit";
    params[ENCODE_RESIDUAL] = true;
  }

  auto algo_params = std::make_unique<raft::spatial::knn::IVFSQParam>();
  algo_params->nlist = Rcpp::as<int>(params[N_LIST]);
  algo_params->nprobe = Rcpp::as<int>(params[N_PROBE]);
  auto const qtype = Rcpp::as<std::string>(params[Q_TYPE]);
  {
    auto const qtype_iter = kQuantizerTypes.find(qtype);
    if (kQuantizerTypes.cend() == qtype_iter) {
      Rcpp::stop("Unsupported quantizer type '" + qtype + "'");
    }
    algo_params->qtype = qtype_iter->second;
  }
  algo_params->encodeResidual = Rcpp::as<bool>(params[ENCODE_RESIDUAL]);

  return algo_params;
}

__host__ std::unique_ptr<raft::spatial::knn::knnIndexParam> build_algo_params(
  KnnAlgo const algo, Rcpp::List const& params, ParamsDetails const& details) {
  bool const automated = (params.size() == 0);

  if (!automated) {
    validate_algo_params(algo, params);
  }

  switch (algo) {
    case KnnAlgo::IVFFLAT:
      return build_ivfflat_algo_params(params, automated);
    case KnnAlgo::IVFPQ:
      return build_ivfpq_algo_params(params, automated, details);
    case KnnAlgo::IVFSQ:
      return build_ivfsq_algo_params(params, automated);
    default:
      return nullptr;
  }
}

}  // namespace

namespace cuml4r {

__host__ SEXP knn_fit(Rcpp::NumericMatrix const& x, int const n_neighbors,
                      int const algo, int const metric, float const p,
                      Rcpp::List const& algo_params) {
  auto const algo_type = static_cast<KnnAlgo>(algo);
  auto const dist_type = static_cast<raft::distance::DistanceType>(metric);

  auto const input_m = cuml4r::Matrix<float>(x, /*transpose=*/false);
  int const n_samples = input_m.numRows;
  int const n_features = input_m.numCols;

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  cuml4r::handle_utils::initializeHandle(handle, stream_view.value());

  // knn input
  auto const& h_x = input_m.values;
  thrust::device_vector<float> d_x(h_x.size());
  auto CUML4R_ANONYMOUS_VARIABLE(x_h2d) = cuml4r::async_copy(
    stream_view.value(), h_x.cbegin(), h_x.cend(), d_x.begin());

  std::unique_ptr<raft::spatial::knn::knnIndex> knn_index(nullptr);

  if (algo_type == KnnAlgo::IVFFLAT || algo_type == KnnAlgo::IVFPQ ||
      algo_type == KnnAlgo::IVFSQ) {
    ParamsDetails details;
    details.numRows_ = n_samples;
    details.numCols_ = n_features;

    auto params =
      build_algo_params(/*algo=*/algo_type, /*params=*/algo_params, details);

    knn_index = std::make_unique<raft::spatial::knn::knnIndex>();
    ML::approx_knn_build_index(handle,
                               /*index=*/knn_index.get(),
                               /*params=*/params.get(),
                               /*metric=*/dist_type,
                               /*metricArg=*/p,
                               /*index_array=*/d_x.data().get(),
                               /*n=*/n_samples,
                               /*D=*/n_features);

    CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));
  }

  Rcpp::List res;
  res["knn_index"] =
    Rcpp::XPtr<raft::spatial::knn::knnIndex>(knn_index.release());

  return res;
}

}  // namespace cuml4r
