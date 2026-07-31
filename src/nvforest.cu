#include "nvforest.h"

#include "cuda_utils.h"
#include "handle_utils.h"
#include "matrix_utils.h"
#include "nvforest_internal.h"
#include "stream_allocator.h"
#include "treelite_utils.cuh"

#include <treelite/c_api.h>
#include <treelite/enum/task_type.h>
#include <treelite/tree.h>
#include <treelite/version.h>
#include <cuml/version_config.hpp>
#include <nvforest/detail/raft_proto/buffer.hpp>
#include <nvforest/forest_model.hpp>
#include <nvforest/infer_kind.hpp>
#include <nvforest/postproc_ops.hpp>
#include <nvforest/tree_layout.hpp>
#include <nvforest/treelite_importer.hpp>
#include <nvforest/version_config.hpp>

#include <Rcpp.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cuml4r {
namespace {

static_assert(CUML_VERSION_MAJOR == 26 && CUML_VERSION_MINOR == 6 &&
              CUML_VERSION_PATCH == 0);
static_assert(NVForest_VERSION_MAJOR == 26 && NVForest_VERSION_MINOR == 6 &&
              NVForest_VERSION_PATCH == 0);
static_assert(TREELITE_VER_MAJOR == 4 && TREELITE_VER_MINOR == 7 &&
              TREELITE_VER_PATCH == 0);

enum class ModelType : int {
  XGBOOST_UBJSON = 0,
  XGBOOST_JSON = 1,
  XGBOOST_LEGACY = 2,
  LIGHTGBM = 3,
  TREELITE_CHECKPOINT = 4
};

enum class PredictionType : int {
  DEFAULT = 0,
  PROBABILITY = 1,
  LEAF_ID = 2,
  PER_TREE = 3
};

SEXP nvforest_model_tag() { return Rf_install("cuda.ml.nvforest_model"); }

struct ModelMetadata {
  treelite::TaskType task_type;
  int num_classes;
  std::string postprocessor;
  bool average_tree_output;
};

class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int const device) {
    CUDA_RT_CALL(cudaGetDevice(&previous_));
    if (previous_ != device) {
      CUDA_RT_CALL(cudaSetDevice(device));
      changed_ = true;
    }
  }

  ScopedCudaDevice(ScopedCudaDevice const&) = delete;
  ScopedCudaDevice& operator=(ScopedCudaDevice const&) = delete;

  ~ScopedCudaDevice() {
    if (changed_) {
      static_cast<void>(cudaSetDevice(previous_));
    }
  }

 private:
  int previous_ = 0;
  bool changed_ = false;
};

bool is_power_of_two(int const value) {
  return value > 0 && (value & (value - 1)) == 0;
}

NvForestOptions parse_options(int const device, int const device_id,
                              int const layout, int const precision,
                              int const default_chunk_size,
                              int const align_bytes) {
  if (device < static_cast<int>(NvForestDevice::CPU) ||
      device > static_cast<int>(NvForestDevice::GPU)) {
    Rcpp::stop("'device' must be either CPU or GPU.");
  }
  if (layout < static_cast<int>(NvForestLayout::DEPTH_FIRST) ||
      layout > static_cast<int>(NvForestLayout::LAYERED)) {
    Rcpp::stop("Unknown nvForest tree layout.");
  }
  if (precision < static_cast<int>(NvForestPrecision::NATIVE) ||
      precision > static_cast<int>(NvForestPrecision::DOUBLE)) {
    Rcpp::stop("Unknown nvForest precision.");
  }
  if (device != static_cast<int>(NvForestDevice::GPU) && device_id != -1) {
    Rcpp::stop("'device_id' is supported only for GPU nvForest models.");
  }
  if (device_id < -1) {
    Rcpp::stop("'device_id' must be NULL or a non-negative integer.");
  }
  if (default_chunk_size < 0) {
    Rcpp::stop("'default_chunk_size' must be NULL or a positive integer.");
  }
  if (device == static_cast<int>(NvForestDevice::GPU) &&
      default_chunk_size != 0 &&
      (!is_power_of_two(default_chunk_size) || default_chunk_size > 32)) {
    Rcpp::stop(
      "GPU 'default_chunk_size' must be a power of two between 1 and 32.");
  }
  if (align_bytes < -1) {
    Rcpp::stop("'align_bytes' must be NULL or a non-negative integer.");
  }

  return NvForestOptions{static_cast<NvForestDevice>(device),
                         device_id,
                         static_cast<NvForestLayout>(layout),
                         static_cast<NvForestPrecision>(precision),
                         default_chunk_size,
                         align_bytes};
}

::nvforest::tree_layout as_nvforest_layout(NvForestLayout const layout) {
  switch (layout) {
    case NvForestLayout::DEPTH_FIRST:
      return ::nvforest::tree_layout::depth_first;
    case NvForestLayout::BREADTH_FIRST:
      return ::nvforest::tree_layout::breadth_first;
    case NvForestLayout::LAYERED:
      return ::nvforest::tree_layout::layered_children_together;
  }
  Rcpp::stop("Unknown nvForest tree layout.");
}

std::optional<bool> as_nvforest_precision(NvForestPrecision const precision) {
  switch (precision) {
    case NvForestPrecision::NATIVE:
      return std::nullopt;
    case NvForestPrecision::SINGLE:
      return false;
    case NvForestPrecision::DOUBLE:
      return true;
  }
  Rcpp::stop("Unknown nvForest precision.");
}

std::optional<::nvforest::index_type> as_chunk_size(int const chunk_size) {
  if (chunk_size == 0) {
    return std::nullopt;
  }
  return static_cast<::nvforest::index_type>(chunk_size);
}

TreeliteHandle load_treelite_model(std::string const& filename,
                                   ModelType const model_type) {
  TreeliteHandle model;
  auto constexpr config = "{}";
  int status = -1;

  switch (model_type) {
    case ModelType::XGBOOST_UBJSON:
      status =
        TreeliteLoadXGBoostModelUBJSON(filename.c_str(), config, model.out());
      break;
    case ModelType::XGBOOST_JSON:
      status =
        TreeliteLoadXGBoostModelJSON(filename.c_str(), config, model.out());
      break;
    case ModelType::XGBOOST_LEGACY:
      status = TreeliteLoadXGBoostModelLegacyBinary(filename.c_str(), config,
                                                    model.out());
      break;
    case ModelType::LIGHTGBM:
      status = TreeliteLoadLightGBMModel(filename.c_str(), config, model.out());
      break;
    case ModelType::TREELITE_CHECKPOINT:
      status = TreeliteDeserializeModelFromFile(filename.c_str(), model.out());
      break;
  }

  treelite_check(status, "Failed to load model '" + filename + "'");
  return model;
}

std::vector<std::uint8_t> serialize_treelite(TreeliteHandle const& model) {
  char const* bytes = nullptr;
  std::size_t size = 0;
  treelite_check(TreeliteSerializeModelToBytes(model.get(), &bytes, &size),
                 "Failed to serialize Treelite model");
  auto const* begin = reinterpret_cast<std::uint8_t const*>(bytes);
  return std::vector<std::uint8_t>(begin, begin + size);
}

TreeliteHandle unserialize_treelite(std::uint8_t const* bytes,
                                    std::size_t const size) {
  if (size == 0) {
    Rcpp::stop("Cannot restore an empty Treelite model.");
  }
  TreeliteHandle model;
  treelite_check(TreeliteDeserializeModelFromBytes(
                   reinterpret_cast<char const*>(bytes), size, model.out()),
                 "Failed to restore Treelite model");
  return model;
}

ModelMetadata read_metadata(TreeliteHandle const& handle) {
  auto const* model = static_cast<treelite::Model const*>(handle.get());
  if (model == nullptr) {
    Rcpp::stop("Treelite returned an empty model.");
  }
  if (model->num_target != 1 || model->num_class.Size() != 1) {
    Rcpp::stop("cuda.ml supports only single-target nvForest models.");
  }

  int num_classes = 0;
  switch (model->task_type) {
    case treelite::TaskType::kBinaryClf:
      num_classes = std::max(2, model->num_class[0]);
      break;
    case treelite::TaskType::kMultiClf:
      num_classes = model->num_class[0];
      if (num_classes < 2) {
        Rcpp::stop(
          "A multiclass Treelite model must contain at least two classes.");
      }
      break;
    case treelite::TaskType::kRegressor:
      break;
    default:
      Rcpp::stop("This Treelite task type is not supported by cuda.ml.");
  }

  return ModelMetadata{model->task_type, num_classes, model->postprocessor,
                       model->average_tree_output};
}

int resolve_device_id(NvForestOptions const& options) {
  if (options.device == NvForestDevice::CPU) {
    return -1;
  }
  if (options.device_id >= 0) {
    return options.device_id;
  }
  return currentDevice();
}

int resolve_align_bytes(NvForestOptions const& options) {
  if (options.align_bytes >= 0) {
    return options.align_bytes;
  }
  return options.device == NvForestDevice::CPU ? 64 : 0;
}

class NvForestModel {
 public:
  NvForestModel(std::unique_ptr<raft::handle_t> handle,
                ::nvforest::forest_model forest,
                std::vector<std::uint8_t> serialized, ModelMetadata metadata,
                NvForestOptions options, int const resolved_device_id,
                bool const averaged_vector_leaf_probabilities)
    : handle_(std::move(handle)),
      forest_(std::move(forest)),
      serialized_(std::move(serialized)),
      metadata_(std::move(metadata)),
      options_(options),
      resolved_device_id_(resolved_device_id),
      averaged_vector_leaf_probabilities_(averaged_vector_leaf_probabilities) {}

  static std::unique_ptr<NvForestModel> create(
    TreeliteHandle const& treelite, NvForestOptions const& options,
    bool const averaged_vector_leaf_probabilities) {
    auto const metadata = read_metadata(treelite);
    if (averaged_vector_leaf_probabilities &&
        (metadata.task_type != treelite::TaskType::kMultiClf ||
         metadata.postprocessor != "identity_multiclass" ||
         !metadata.average_tree_output)) {
      Rcpp::stop(
        "cuML random forest returned an unexpected Treelite probability "
        "contract.");
    }
    auto serialized = serialize_treelite(treelite);
    auto const device_id = resolve_device_id(options);
    auto const device_type = options.device == NvForestDevice::GPU
                               ? raft_proto::device_type::gpu
                               : raft_proto::device_type::cpu;
    auto handle = std::unique_ptr<raft::handle_t>();
    auto stream = cudaStream_t{};
    auto device_guard = std::unique_ptr<ScopedCudaDevice>();

    if (options.device == NvForestDevice::GPU) {
      device_guard = std::make_unique<ScopedCudaDevice>(device_id);
      auto const stream_view = stream_allocator::getOrCreateStream();
      handle = std::make_unique<raft::handle_t>();
      handle_utils::initializeHandle(*handle, stream_view.value());
      stream = stream_view.value();
    }

    auto forest = ::nvforest::import_from_treelite_handle(
      treelite.get(), as_nvforest_layout(options.layout),
      static_cast<::nvforest::index_type>(resolve_align_bytes(options)),
      as_nvforest_precision(options.precision), device_type,
      options.device == NvForestDevice::GPU ? device_id : 0, stream);

    if (handle != nullptr) {
      handle->sync_stream();
    }

    return std::make_unique<NvForestModel>(
      std::move(handle), std::move(forest), std::move(serialized), metadata,
      options, device_id, averaged_vector_leaf_probabilities);
  }

  bool is_classifier() const {
    return metadata_.task_type == treelite::TaskType::kBinaryClf ||
           metadata_.task_type == treelite::TaskType::kMultiClf;
  }

  std::size_t output_count(::nvforest::infer_kind const kind) {
    if (kind == ::nvforest::infer_kind::leaf_id) {
      return forest_.num_trees();
    }
    if (kind == ::nvforest::infer_kind::per_tree) {
      return forest_.num_trees() *
             (forest_.has_vector_leaves() ? forest_.num_outputs() : 1);
    }
    return forest_.num_outputs();
  }

  template <typename T>
  std::vector<T> infer(Rcpp::NumericMatrix const& input,
                       ::nvforest::infer_kind const kind,
                       int const chunk_size) {
    auto const matrix = Matrix<T>(input, /*transpose=*/false);
    auto const output_size = matrix.numRows * output_count(kind);
    auto const effective_chunk_size =
      chunk_size == 0 ? options_.default_chunk_size : chunk_size;
    validate_chunk_size(effective_chunk_size);

    if (options_.device == NvForestDevice::CPU) {
      auto input_buffer = raft_proto::buffer<T>(
        const_cast<T*>(matrix.values.data()), matrix.values.size(),
        raft_proto::device_type::cpu, 0);
      auto output = std::vector<T>(output_size);
      auto output_buffer = raft_proto::buffer<T>(
        output.data(), output.size(), raft_proto::device_type::cpu, 0);
      forest_.predict(output_buffer, input_buffer, cudaStream_t{}, kind,
                      as_chunk_size(effective_chunk_size));
      return output;
    }

    ScopedCudaDevice const device_guard(resolved_device_id_);
    auto const stream = handle_->get_stream();
    auto input_buffer = raft_proto::buffer<T>(
      matrix.values.cbegin(), matrix.values.cend(),
      raft_proto::device_type::gpu, resolved_device_id_, stream);
    auto output_buffer = raft_proto::buffer<T>(
      output_size, raft_proto::device_type::gpu, resolved_device_id_, stream);
    forest_.predict(output_buffer, input_buffer, stream, kind,
                    as_chunk_size(effective_chunk_size));
    auto host_output = raft_proto::buffer<T>(
      output_buffer, raft_proto::device_type::cpu, 0, stream);
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    return std::vector<T>(host_output.data(),
                          host_output.data() + host_output.size());
  }

  void validate_input(Rcpp::NumericMatrix const& input) {
    if (input.ncol() != forest_.num_features()) {
      Rcpp::stop("nvForest model expects %d features, but received %d.",
                 static_cast<int>(forest_.num_features()), input.ncol());
    }
  }

  void validate_chunk_size(int const chunk_size) const {
    if (chunk_size == 0) {
      return;
    }
    if (options_.device == NvForestDevice::GPU &&
        (!is_power_of_two(chunk_size) || chunk_size > 32)) {
      Rcpp::stop("GPU 'chunk_size' must be a power of two between 1 and 32.");
    }
  }

  bool has_probability_output() const {
    return is_classifier() && (metadata_.postprocessor == "sigmoid" ||
                               metadata_.postprocessor == "softmax" ||
                               metadata_.postprocessor == "multiclass_ova" ||
                               averaged_vector_leaf_probabilities_);
  }

  void validate_class_support() const {
    if (has_probability_output() || metadata_.postprocessor == "max_index" ||
        metadata_.postprocessor == "hinge") {
      return;
    }
    Rcpp::stop(
      "Class prediction is not supported for Treelite postprocessor '%s'.",
      metadata_.postprocessor.c_str());
  }

  void validate_probability_support() const {
    if (!has_probability_output()) {
      Rcpp::stop(
        "Probability prediction is not supported for Treelite postprocessor "
        "'%s'.",
        metadata_.postprocessor.c_str());
    }
  }

  std::unique_ptr<raft::handle_t> handle_;
  // This member must be destroyed before handle_.
  ::nvforest::forest_model forest_;
  std::vector<std::uint8_t> serialized_;
  ModelMetadata metadata_;
  NvForestOptions options_;
  int resolved_device_id_;
  bool averaged_vector_leaf_probabilities_;
};

ModelType infer_model_type(std::string const& filename) {
  auto const dot = filename.find_last_of('.');
  if (dot == std::string::npos) {
    Rcpp::stop("Cannot infer model type from a filename without a suffix.");
  }
  auto suffix = filename.substr(dot);
  std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                 [](unsigned char const value) {
                   return static_cast<char>(std::tolower(value));
                 });
  if (suffix == ".ubj") {
    return ModelType::XGBOOST_UBJSON;
  }
  if (suffix == ".json") {
    return ModelType::XGBOOST_JSON;
  }
  if (suffix == ".model") {
    return ModelType::XGBOOST_LEGACY;
  }
  if (suffix == ".txt") {
    return ModelType::LIGHTGBM;
  }
  Rcpp::stop("Cannot infer nvForest model type from suffix '%s'.",
             suffix.c_str());
}

template <typename T>
Rcpp::NumericMatrix numeric_matrix(std::vector<T> const& values,
                                   int const n_rows, int const n_cols) {
  Rcpp::NumericMatrix output(n_rows, n_cols);
  for (int row = 0; row < n_rows; ++row) {
    for (int col = 0; col < n_cols; ++col) {
      output(row, col) = values[row * n_cols + col];
    }
  }
  return output;
}

template <typename T>
SEXP default_predictions(NvForestModel& model, std::vector<T> const& values,
                         int const n_rows, double const threshold) {
  auto const n_outputs =
    model.output_count(::nvforest::infer_kind::default_kind);
  if (!model.is_classifier()) {
    if (n_outputs != 1) {
      Rcpp::stop("Only single-output nvForest regression is supported.");
    }
    return Rcpp::NumericVector(values.begin(), values.end());
  }

  if (threshold < 0.0 || threshold > 1.0) {
    Rcpp::stop("'threshold' must lie in [0, 1].");
  }
  Rcpp::IntegerVector output(n_rows);
  auto const row_op = model.forest_.row_postprocessing();
  for (int row = 0; row < n_rows; ++row) {
    auto const begin = values.begin() + row * n_outputs;
    if (row_op == ::nvforest::row_op::max_index) {
      output[row] = static_cast<int>(std::llround(begin[0]));
    } else if (model.metadata_.num_classes == 2 && n_outputs == 1) {
      output[row] = static_cast<int>(begin[0] >= threshold);
    } else if (model.metadata_.num_classes == 2 && n_outputs == 2) {
      output[row] = static_cast<int>(begin[1] >= threshold);
    } else {
      output[row] = static_cast<int>(
        std::distance(begin, std::max_element(begin, begin + n_outputs)));
    }
  }
  return output;
}

template <typename T>
Rcpp::NumericMatrix probability_predictions(NvForestModel& model,
                                            std::vector<T> const& values,
                                            int const n_rows) {
  auto const n_outputs =
    model.output_count(::nvforest::infer_kind::default_kind);
  auto const n_classes = model.metadata_.num_classes;
  if (n_outputs == static_cast<std::size_t>(n_classes)) {
    return numeric_matrix(values, n_rows, n_classes);
  }
  if (n_outputs == 1 && n_classes == 2) {
    Rcpp::NumericMatrix output(n_rows, 2);
    for (int row = 0; row < n_rows; ++row) {
      output(row, 1) = values[row];
      output(row, 0) = 1.0 - output(row, 1);
    }
    return output;
  }
  Rcpp::stop("nvForest returned %d outputs for a %d-class model.",
             static_cast<int>(n_outputs), n_classes);
}

template <typename T>
Rcpp::IntegerMatrix leaf_predictions(std::vector<T> const& values,
                                     int const n_rows, int const n_cols) {
  Rcpp::IntegerMatrix output(n_rows, n_cols);
  for (int row = 0; row < n_rows; ++row) {
    for (int col = 0; col < n_cols; ++col) {
      auto const value = static_cast<double>(values[row * n_cols + col]);
      auto const rounded = std::round(value);
      if (!std::isfinite(value) || value != rounded || rounded < 0 ||
          rounded > std::numeric_limits<int>::max()) {
        Rcpp::stop("nvForest returned an invalid leaf ID.");
      }
      output(row, col) = static_cast<int>(rounded);
    }
  }
  return output;
}

SEXP predict_impl(NvForestModel& model, Rcpp::NumericMatrix const& input,
                  PredictionType const prediction_type, double const threshold,
                  int const chunk_size) {
  model.validate_input(input);
  auto const n_rows = input.nrow();
  auto kind = ::nvforest::infer_kind::default_kind;
  if (prediction_type == PredictionType::DEFAULT && model.is_classifier()) {
    model.validate_class_support();
  } else if (prediction_type == PredictionType::PROBABILITY) {
    if (!model.is_classifier()) {
      Rcpp::stop("Probability prediction requires a classifier.");
    }
    model.validate_probability_support();
  } else if (prediction_type == PredictionType::LEAF_ID) {
    kind = ::nvforest::infer_kind::leaf_id;
  } else if (prediction_type == PredictionType::PER_TREE) {
    kind = ::nvforest::infer_kind::per_tree;
  }

  auto const n_outputs = static_cast<int>(model.output_count(kind));
  if (model.forest_.is_double_precision()) {
    auto const values = model.infer<double>(input, kind, chunk_size);
    switch (prediction_type) {
      case PredictionType::DEFAULT:
        return default_predictions(model, values, n_rows, threshold);
      case PredictionType::PROBABILITY:
        return probability_predictions(model, values, n_rows);
      case PredictionType::LEAF_ID:
        return leaf_predictions(values, n_rows, n_outputs);
      case PredictionType::PER_TREE:
        return numeric_matrix(values, n_rows, n_outputs);
    }
  } else {
    auto const values = model.infer<float>(input, kind, chunk_size);
    switch (prediction_type) {
      case PredictionType::DEFAULT:
        return default_predictions(model, values, n_rows, threshold);
      case PredictionType::PROBABILITY:
        return probability_predictions(model, values, n_rows);
      case PredictionType::LEAF_ID:
        return leaf_predictions(values, n_rows, n_outputs);
      case PredictionType::PER_TREE:
        return numeric_matrix(values, n_rows, n_outputs);
    }
  }
  Rcpp::stop("Unknown nvForest prediction type.");
}

NvForestModel& as_model(SEXP model) {
  if (TYPEOF(model) != EXTPTRSXP ||
      R_ExternalPtrTag(model) != nvforest_model_tag()) {
    Rcpp::stop("Expected an nvForest model pointer.");
  }
  auto pointer = Rcpp::XPtr<NvForestModel>(model);
  if (pointer.get() == nullptr) {
    Rcpp::stop("nvForest model pointer is null.");
  }
  return *pointer;
}

}  // namespace

SEXP nvforest_from_treelite(TreeliteHandle&& treelite,
                            NvForestOptions const& options,
                            bool const averaged_vector_leaf_probabilities) {
  auto model = NvForestModel::create(treelite, options,
                                     averaged_vector_leaf_probabilities);
  return Rcpp::XPtr<NvForestModel>(model.release(), true, nvforest_model_tag());
}

SEXP nvforest_load_model(std::string const& filename, int const model_type,
                         int const device, int const device_id,
                         int const layout, int const precision,
                         int const default_chunk_size, int const align_bytes) {
  if (model_type < -1 ||
      model_type > static_cast<int>(ModelType::TREELITE_CHECKPOINT)) {
    Rcpp::stop("Unknown nvForest model type.");
  }
  auto const type = model_type == -1 ? infer_model_type(filename)
                                     : static_cast<ModelType>(model_type);
  auto treelite = load_treelite_model(filename, type);
  auto const options = parse_options(device, device_id, layout, precision,
                                     default_chunk_size, align_bytes);
  return nvforest_from_treelite(std::move(treelite), options,
                                /*averaged_vector_leaf_probabilities=*/false);
}

Rcpp::List nvforest_model_info(SEXP model) {
  auto& value = as_model(model);
  return Rcpp::List::create(
    Rcpp::Named("task_type") = static_cast<int>(value.metadata_.task_type),
    Rcpp::Named("num_classes") = value.metadata_.num_classes,
    Rcpp::Named("num_features") =
      static_cast<int>(value.forest_.num_features()),
    Rcpp::Named("num_outputs") = static_cast<int>(value.forest_.num_outputs()),
    Rcpp::Named("num_trees") = static_cast<int>(value.forest_.num_trees()),
    Rcpp::Named("has_vector_leaves") = value.forest_.has_vector_leaves(),
    Rcpp::Named("average_tree_output") = value.metadata_.average_tree_output,
    Rcpp::Named("has_probability_output") = value.has_probability_output(),
    Rcpp::Named("device") = static_cast<int>(value.options_.device),
    Rcpp::Named("device_id") = value.resolved_device_id_,
    Rcpp::Named("layout") = static_cast<int>(value.options_.layout),
    Rcpp::Named("precision") = value.forest_.is_double_precision() ? 1 : 0,
    Rcpp::Named("default_chunk_size") = value.options_.default_chunk_size,
    Rcpp::Named("align_bytes") = resolve_align_bytes(value.options_),
    Rcpp::Named("treelite_postprocessor") = value.metadata_.postprocessor);
}

SEXP nvforest_predict(SEXP model, Rcpp::NumericMatrix const& input,
                      int const prediction_type, double const threshold,
                      int const chunk_size) {
  if (prediction_type < static_cast<int>(PredictionType::DEFAULT) ||
      prediction_type > static_cast<int>(PredictionType::PER_TREE)) {
    Rcpp::stop("Unknown nvForest prediction type.");
  }
  if (chunk_size < 0) {
    Rcpp::stop("'chunk_size' must be NULL or a positive integer.");
  }
  return predict_impl(as_model(model), input,
                      static_cast<PredictionType>(prediction_type), threshold,
                      chunk_size);
}

Rcpp::RawVector nvforest_serialize(SEXP model) {
  auto const& bytes = as_model(model).serialized_;
  Rcpp::RawVector output(bytes.size());
  std::copy(bytes.cbegin(), bytes.cend(), output.begin());
  return output;
}

SEXP nvforest_unserialize(Rcpp::RawVector const& bytes, int const device,
                          int const device_id, int const layout,
                          int const precision, int const default_chunk_size,
                          int const align_bytes,
                          bool const averaged_vector_leaf_probabilities) {
  auto treelite = unserialize_treelite(bytes.begin(), bytes.size());
  auto const options = parse_options(device, device_id, layout, precision,
                                     default_chunk_size, align_bytes);
  return nvforest_from_treelite(std::move(treelite), options,
                                averaged_vector_leaf_probabilities);
}

}  // namespace cuml4r
