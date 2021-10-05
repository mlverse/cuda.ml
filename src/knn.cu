#include "async_utils.cuh"
#include "cuda_utils.h"
#include "handle_utils.h"
#include "knn_detail.h"
#include "matrix_utils.h"
#include "pinned_host_vector.h"
#include "preprocessor.h"
#include "random_forest.cuh"
#include "stream_allocator.h"

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <cuml/neighbors/knn.hpp>
#include <cuml/version_config.hpp>

#include <Rcpp.h>

#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#if CUML_VERSION_MAJOR == 21
#if CUML4R_CONCAT(0x, CUML_VERSION_MINOR) >= 0x08

#include <raft/spatial/knn/ann_common.h>

using knnIndex = raft::spatial::knn::knnIndex;
using knnIndexParam = raft::spatial::knn::knnIndexParam;
using QuantizerType = raft::spatial::knn::QuantizerType;
using IVFFlatParam = raft::spatial::knn::IVFFlatParam;
using IVFPQParam = raft::spatial::knn::IVFPQParam;
using IVFSQParam = raft::spatial::knn::IVFSQParam;

#else

using knnIndex = ML::knnIndex;
using knnIndexParam = ML::knnIndexParam;
using QuantizerType = ML::QuantizerType;
using IVFFlatParam = ML::IVFFlatParam;
using IVFPQParam = ML::IVFPQParam;
using IVFSQParam = ML::IVFSQParam;

#endif
#endif

namespace cuml4r {
namespace knn {
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

std::unordered_map<std::string, QuantizerType> const
  kQuantizerTypes{
    {"QT_8bit", QuantizerType::QT_8bit},
    {"QT_4bit", QuantizerType::QT_4bit},
    {"QT_8bit_uniform", QuantizerType::QT_8bit_uniform},
    {"QT_4bit_uniform", QuantizerType::QT_4bit_uniform},
    {"QT_fp16", QuantizerType::QT_fp16},
    {"QT_8bit_direct", QuantizerType::QT_8bit_direct},
    {"QT_6bit", QuantizerType::QT_6bit}};

// Additional info for setting KNN params
struct ParamsDetails {
  int numRows_;
  int numCols_;
};

enum class Algo { BRUTE_FORCE = 0, IVFFLAT = 1, IVFPQ = 2, IVFSQ = 3 };

struct NearestNeighbors {
  NearestNeighbors() {}
  NearestNeighbors(int const n_samples, int const n_neighbors) {
    auto const n_entries = n_samples * n_neighbors;
    indices.resize(n_entries);
    dists.resize(n_entries);
  }

  thrust::device_vector<int64_t> indices;
  thrust::device_vector<float> dists;
};

template <typename ResponseT>
class PredictionCtx {
 public:
  using ResponseVecT = typename knn::detail::RcppVector<ResponseT>::type;
  __host__ PredictionCtx(Rcpp::List const& model, Rcpp::NumericMatrix const& x,
                         int const n_neighbors)
    : nSamples_(x.nrow()),
      nFeatures_(x.ncol()),
      modelKnnIndex_(Rcpp::XPtr<knnIndex>(
        static_cast<SEXP>(model[KNN_INDEX]))),
      modelAlgoType_(static_cast<knn::Algo>(Rcpp::as<int>(model[ALGO]))),
      modelDistType_(static_cast<raft::distance::DistanceType>(
        Rcpp::as<int>(model[METRIC]))),
      modelP_(Rcpp::as<float>(model[P_VALUE])),
      modelNSamples_(Rcpp::as<int>(model[N_SAMPLES])),
      modelNDims_(Rcpp::as<int>(model[N_DIMS])),
      streamView_(cuml4r::stream_allocator::getOrCreateStream()) {
    cuml4r::handle_utils::initializeHandle(handle_, streamView_.value());
    auto const input_m = cuml4r::Matrix<float>(x, /*transpose=*/false);
    // KNN classifier input
    auto const& h_x = input_m.values;
    dX_.resize(h_x.size());
    xH2D_ = cuml4r::async_copy(streamView_.value(), h_x.cbegin(), h_x.cend(),
                               dX_.begin());

    ResponseVecT const model_resps(
      Rcpp::as<ResponseVecT>(model[detail::kResponses]));
    auto h_y = Rcpp::as<cuml4r::pinned_host_vector<ResponseT>>(model_resps);
    dY_.resize(h_y.size());
    yH2D_ = cuml4r::async_copy(streamView_.value(), h_y.cbegin(), h_y.cend(),
                               dY_.begin());

    nearestNeighbors_ = query_nearest_neighbors(n_neighbors);

    CUDA_RT_CALL(cudaStreamSynchronize(streamView_.value()));
  }

  __host__ NearestNeighbors query_nearest_neighbors(int const n_neighbors) {
    NearestNeighbors res(nSamples_, n_neighbors);
    auto d_input = dX_.data().get();

    if (modelAlgoType_ == Algo::BRUTE_FORCE) {
      std::vector<float*> input{d_input};
      std::vector<int> sizes{nSamples_};

      ML::brute_force_knn(handle_, input, sizes, /*D=*/modelNDims_,
                          /*search_items=*/d_input,
                          /*n=*/nSamples_, /*res_I=*/res.indices.data().get(),
                          /*res_D=*/res.dists.data().get(), /*k=*/n_neighbors,
                          /*rowMajorIndex=*/true, /*rowMajorQuery=*/true,
                          /*metric=*/modelDistType_, modelP_);
    } else {
      ML::approx_knn_search(handle_, /*distances=*/res.dists.data().get(),
                            /*indices=*/res.indices.data().get(),
                            /*index=*/modelKnnIndex_.get(), /*k=*/n_neighbors,
                            /*query_array=*/d_input, /*n=*/nSamples_);
    }

    return res;
  }

  // input dimensions
  int const nSamples_;
  int const nFeatures_;
  // attributes from the KNN model object
  Rcpp::XPtr<knnIndex> const modelKnnIndex_;
  Algo const modelAlgoType_;
  raft::distance::DistanceType const modelDistType_;
  float const modelP_;
  int const modelNSamples_;
  int const modelNDims_;
  // CUDA stream, etc
  rmm::cuda_stream_view streamView_;
  raft::handle_t handle_;
  // KNN classifier inputs
  thrust::device_vector<float> dX_;
  thrust::device_vector<ResponseT> dY_;
  NearestNeighbors nearestNeighbors_;

 private:
  cuml4r::unique_marker xH2D_;
  cuml4r::unique_marker yH2D_;
};

__host__ void validate_param_list(
  Rcpp::List const& params, std::initializer_list<char const*> required_attrs) {
  for (auto const attr : required_attrs) {
    if (!params.containsElementNamed(attr)) {
      Rcpp::stop("Required attribute '%s' is missing from algo params!", attr);
    }
  }
}

__host__ void validate_algo_params(Algo const algo, Rcpp::List const& params) {
  if (algo == Algo::IVFFLAT) {
    validate_param_list(params, {N_LIST, N_PROBE});
  } else if (algo == Algo::IVFPQ) {
    validate_param_list(
      params, {N_LIST, N_PROBE, M_VALUE, N_BITS, USE_PRE_COMPUTED_TABLES});
  } else if (algo == Algo::IVFSQ) {
    validate_param_list(params, {N_LIST, N_PROBE, Q_TYPE, ENCODE_RESIDUAL});
  }
}

__host__ std::unique_ptr<knnIndexParam>
build_ivfflat_algo_params(Rcpp::List params, bool const automated) {
  if (automated) {
    params[N_LIST] = 8;
    params[N_PROBE] = 2;
  }

  auto algo_params = std::make_unique<IVFFlatParam>();
  algo_params->nlist = params[N_LIST];
  algo_params->nprobe = params[N_PROBE];

  return algo_params;
}

__host__ std::unique_ptr<knnIndexParam>
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

  auto algo_params = std::make_unique<IVFPQParam>();
  algo_params->nlist = Rcpp::as<int>(params[N_LIST]);
  algo_params->nprobe = Rcpp::as<int>(params[N_PROBE]);
  algo_params->M = Rcpp::as<int>(params[M_VALUE]);
  algo_params->n_bits = Rcpp::as<int>(params[N_BITS]);
  algo_params->usePrecomputedTables =
    Rcpp::as<bool>(params[USE_PRE_COMPUTED_TABLES]);

  return algo_params;
}

__host__ std::unique_ptr<knnIndexParam>
build_ivfsq_algo_params(Rcpp::List params, bool const automated) {
  if (automated) {
    params[N_LIST] = 8;
    params[N_PROBE] = 2;
    params[Q_TYPE] = "QT_8bit";
    params[ENCODE_RESIDUAL] = true;
  }

  auto algo_params = std::make_unique<IVFSQParam>();
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

__host__ std::unique_ptr<knnIndexParam> build_algo_params(
  Algo const algo, Rcpp::List const& params, ParamsDetails const& details) {
  bool const automated = (params.size() == 0);

  if (!automated) {
    validate_algo_params(algo, params);
  }

  switch (algo) {
    case Algo::IVFFLAT:
      return build_ivfflat_algo_params(params, automated);
    case Algo::IVFPQ:
      return build_ivfpq_algo_params(params, automated, details);
    case Algo::IVFSQ:
      return build_ivfsq_algo_params(params, automated);
    default:
      return nullptr;
  }
}

__host__ std::unique_ptr<knnIndex> build_knn_index(
  raft::handle_t& handle, float* const d_input, int const n_samples,
  int const n_features, Algo const algo_type,
  raft::distance::DistanceType const dist_type, float const p,
  Rcpp::List const& algo_params) {
  std::unique_ptr<knnIndex> knn_index(nullptr);

  if (algo_type == Algo::IVFFLAT || algo_type == Algo::IVFPQ ||
      algo_type == Algo::IVFSQ) {
    ParamsDetails details;
    details.numRows_ = n_samples;
    details.numCols_ = n_features;

    auto params =
      build_algo_params(/*algo=*/algo_type, /*params=*/algo_params, details);

    knn_index = std::make_unique<knnIndex>();
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

}  // namespace
}  // namespace knn

__host__ Rcpp::List knn_fit(Rcpp::NumericMatrix const& x, int const algo,
                            int const metric, float const p,
                            Rcpp::List const& algo_params) {
  auto const algo_type = static_cast<knn::Algo>(algo);
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
  model[knn::KNN_INDEX] =
    Rcpp::XPtr<knnIndex>(knn_index.release());
  model[knn::ALGO] = algo;
  model[knn::METRIC] = metric;
  model[knn::P_VALUE] = p;
  model[knn::N_SAMPLES] = n_samples;
  model[knn::N_DIMS] = n_features;

  return model;
}

__host__ Rcpp::IntegerVector knn_classifier_predict(
  Rcpp::List const& model, Rcpp::NumericMatrix const& x,
  int const n_neighbors) {
  // KNN classifier input & pre-processing
  knn::PredictionCtx<int> ctx(model, x, n_neighbors);
  std::vector<int*> y_vec{ctx.dY_.data().get()};

  // KNN classifier output
  thrust::device_vector<int> d_out(ctx.nSamples_);

  ML::knn_classify(/*handle=*/ctx.handle_, /*out=*/d_out.data().get(),
                   /*knn_indices=*/ctx.nearestNeighbors_.indices.data().get(),
                   /*y=*/y_vec, /*n_index_rows=*/ctx.modelNSamples_,
                   /*n_query_rows=*/ctx.nSamples_, /*k=*/n_neighbors);

  cuml4r::pinned_host_vector<int> h_out(d_out.size());
  auto CUML4R_ANONYMOUS_VARIABLE(out_d2h) = cuml4r::async_copy(
    ctx.streamView_.value(), d_out.cbegin(), d_out.cend(), h_out.begin());
  CUDA_RT_CALL(cudaStreamSynchronize(ctx.streamView_.value()));

  return Rcpp::IntegerVector(h_out.cbegin(), h_out.cend());
}

__host__ Rcpp::NumericMatrix knn_classifier_predict_probabilities(
  Rcpp::List const& model, Rcpp::NumericMatrix const& x,
  int const n_neighbors) {
  // KNN classifier input & pre-processing
  knn::PredictionCtx<int> ctx(model, x, n_neighbors);
  std::vector<int*> y_vec{ctx.dY_.data().get()};
  int const n_classes =
    Rcpp::unique(Rcpp::as<Rcpp::IntegerVector>(model[knn::detail::kResponses]))
      .size();

  // KNN classifier output
  thrust::device_vector<float> d_out(ctx.nSamples_ * n_classes);
  std::vector<float*> out_vec{d_out.data().get()};

  ML::knn_class_proba(
    /*handle=*/ctx.handle_, /*out=*/out_vec,
    /*knn_indices=*/ctx.nearestNeighbors_.indices.data().get(),
    /*y=*/y_vec, /*n_index_rows=*/ctx.modelNSamples_,
    /*n_query_rows=*/ctx.nSamples_, /*k=*/n_neighbors);

  cuml4r::pinned_host_vector<float> h_out(d_out.size());
  auto CUML4R_ANONYMOUS_VARIABLE(out_d2h) = cuml4r::async_copy(
    ctx.streamView_.value(), d_out.cbegin(), d_out.cend(), h_out.begin());
  CUDA_RT_CALL(cudaStreamSynchronize(ctx.streamView_.value()));

  return Rcpp::transpose(
    Rcpp::NumericMatrix(n_classes, ctx.nSamples_, h_out.begin()));
}

Rcpp::NumericVector knn_regressor_predict(Rcpp::List const& model,
                                          Rcpp::NumericMatrix const& x,
                                          int const n_neighbors) {
  // KNN regressor input & pre-processing
  knn::PredictionCtx<float> ctx(model, x, n_neighbors);
  std::vector<float*> y_vec{ctx.dY_.data().get()};

  // KNN regressor output
  thrust::device_vector<float> d_out(ctx.nSamples_);

  ML::knn_regress(/*handle=*/ctx.handle_, /*out=*/d_out.data().get(),
                  /*knn_indices=*/ctx.nearestNeighbors_.indices.data().get(),
                  /*y=*/y_vec,
                  /*n_rows=*/ctx.modelNSamples_,
                  /*n_samples=*/ctx.nSamples_, /*k=*/n_neighbors);

  cuml4r::pinned_host_vector<float> h_out(d_out.size());
  auto CUML4R_ANONYMOUS_VARIABLE(out_d2h) = cuml4r::async_copy(
    ctx.streamView_.value(), d_out.cbegin(), d_out.cend(), h_out.begin());
  CUDA_RT_CALL(cudaStreamSynchronize(ctx.streamView_.value()));

  return Rcpp::NumericVector(h_out.begin(), h_out.end());
}

}  // namespace cuml4r
