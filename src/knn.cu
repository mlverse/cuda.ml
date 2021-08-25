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

// string constants related to KNN params
char const* const N_LIST = "nlist";
char const* const N_PROBE = "nprobe";
char const* const M_VALUE = "M";
char const* const N_BITS = "n_bits";
char const* const USE_PRE_COMPUTED_TABLES = "usePrecomputedTables";
char const* const Q_TYPE = "qtype";
char const* const ENCODE_RESIDUAL = "encodeResidual";
// string constants related to KNN model attributes
char const* const KNN_INDEX = "knn_index";
char const* const ALGO = "algo";
char const* const P_VALUE = "p";
char const* const METRIC = "metric";
char const* const N_SAMPLES = "n_samples";
char const* const N_DIMS = "n_dims";
char const* const LABELS = "labels";

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

struct KnnQueryResult {
  KnnQueryResult(int const n_samples, int const n_neighbors) {
    auto const n_entries = n_samples * n_neighbors;
    indices.resize(n_entries);
    dists.resize(n_entries);
  }

  thrust::device_vector<int64_t> indices;
  thrust::device_vector<float> dists;
};

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

__host__ std::unique_ptr<raft::spatial::knn::knnIndex> build_knn_index(
  raft::handle_t& handle, float* const d_input, int const n_samples,
  int const n_features, KnnAlgo const algo_type,
  raft::distance::DistanceType const dist_type, float const p,
  Rcpp::List const& algo_params) {
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
                               /*index_array=*/d_input,
                               /*n=*/n_samples,
                               /*D=*/n_features);

    CUDA_RT_CALL(cudaStreamSynchronize(handle.get_stream()));
  }

  return knn_index;
}

__host__ KnnQueryResult query_nearest_neighbors(
  raft::handle_t& handle, float* const d_input,
  raft::spatial::knn::knnIndex* const knn_index, KnnAlgo const algo_type,
  raft::distance::DistanceType const dist_type, float const p,
  int const n_samples, int const n_dims, int const n_neighbors) {
  KnnQueryResult res(n_samples, n_neighbors);

  if (algo_type == KnnAlgo::BRUTE_FORCE) {
    std::vector<float*> input{d_input};
    std::vector<int> sizes{n_samples};

    ML::brute_force_knn(
      handle, input, sizes, /*D=*/n_dims, /*search_items=*/d_input,
      /*n=*/n_samples, /*res_I=*/res.indices.data().get(),
      /*res_D=*/res.dists.data().get(), /*k=*/n_neighbors,
      /*rowMajorIndex=*/true, /*rowMajorQuery=*/true, /*metric=*/dist_type, p);
  } else {
    ML::approx_knn_search(handle, /*distances=*/res.dists.data().get(),
                          /*indices=*/res.indices.data().get(),
                          /*index=*/knn_index, /*k=*/n_neighbors,
                          /*query_array=*/d_input, /*n=*/n_samples);
  }

  return res;
}

}  // namespace

namespace cuml4r {

__host__ SEXP knn_fit(Rcpp::NumericMatrix const& x,
                      Rcpp::IntegerVector const& y, int const algo,
                      int const metric, float const p,
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

  auto knn_index =
    build_knn_index(handle, /*d_input=*/d_x.data().get(), n_samples, n_features,
                    algo_type, dist_type, p, algo_params);

  Rcpp::List model;
  model[KNN_INDEX] =
    Rcpp::XPtr<raft::spatial::knn::knnIndex>(knn_index.release());
  model[ALGO] = algo;
  model[METRIC] = metric;
  model[P_VALUE] = p;
  model[N_SAMPLES] = n_samples;
  model[N_DIMS] = n_features;
  model[LABELS] = y;

  return model;
}

__host__ Rcpp::IntegerVector knn_classifier_predict(
  Rcpp::List const& model, Rcpp::NumericMatrix const& x,
  int const n_neighbors) {
  auto const model_knn_index = Rcpp::XPtr<raft::spatial::knn::knnIndex>(
    static_cast<SEXP>(model[KNN_INDEX]));
  auto const model_algo_type = static_cast<KnnAlgo>(Rcpp::as<int>(model[ALGO]));
  auto const model_dist_type =
    static_cast<raft::distance::DistanceType>(Rcpp::as<int>(model[METRIC]));
  auto const model_p = Rcpp::as<float>(model[P_VALUE]);
  auto const model_n_samples = Rcpp::as<int>(model[N_SAMPLES]);
  auto const model_n_dims = Rcpp::as<int>(model[N_DIMS]);
  auto const model_labels = Rcpp::as<Rcpp::IntegerVector>(model[LABELS]);

  auto const input_m = cuml4r::Matrix<float>(x, /*transpose=*/false);

  auto stream_view = cuml4r::stream_allocator::getOrCreateStream();
  raft::handle_t handle;
  cuml4r::handle_utils::initializeHandle(handle, stream_view.value());

  // KNN classifier input
  auto const& h_x = input_m.values;
  thrust::device_vector<float> d_x(h_x.size());
  auto CUML4R_ANONYMOUS_VARIABLE(x_h2d) = cuml4r::async_copy(
    stream_view.value(), h_x.cbegin(), h_x.cend(), d_x.begin());
  auto h_y = Rcpp::as<cuml4r::pinned_host_vector<int>>(model_labels);
  thrust::device_vector<int> d_y(h_y.size());
  auto CUML4R_ANONYMOUS_VARIABLE(y_h2d) = cuml4r::async_copy(
    stream_view.value(), h_y.cbegin(), h_y.cend(), d_y.begin());
  std::vector<int*> y_vec{d_y.data().get()};

  // KNN classifier output
  thrust::device_vector<int> d_out(input_m.numRows);

  auto knn_query_res = query_nearest_neighbors(
    handle, /*d_input=*/d_x.data().get(), /*knn_index=*/model_knn_index.get(),
    /*algo_type=*/model_algo_type,
    /*dist_type=*/model_dist_type, /*p=*/model_p, /*n_samples=*/input_m.numRows,
    /*ndims=*/model_n_dims, n_neighbors);
  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  ML::knn_classify(handle, /*out=*/d_out.data().get(),
                   /*knn_indices=*/knn_query_res.indices.data().get(),
                   /*y=*/y_vec, /*n_index_rows=*/model_n_samples,
                   /*n_query_rows=*/input_m.numRows, /*k=*/n_neighbors);

  cuml4r::pinned_host_vector<int> h_out(d_out.size());
  auto CUML4R_ANONYMOUS_VARIABLE(out_d2h) = cuml4r::async_copy(
    stream_view.value(), d_out.cbegin(), d_out.cend(), h_out.begin());
  CUDA_RT_CALL(cudaStreamSynchronize(stream_view.value()));

  return Rcpp::IntegerVector(h_out.cbegin(), h_out.cend());
}

}  // namespace cuml4r
