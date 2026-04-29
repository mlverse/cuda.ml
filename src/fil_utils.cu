#include "fil_utils.h"

#ifndef CUML4R_TREELITE_C_API_MISSING

#include "cuda_utils.h"

#include <Rcpp.h>

#include <cuml/fil/detail/raft_proto/handle.hpp>
#include <cuml/fil/detail/device_initialization/gpu.cuh>
#include <cuml/fil/detail/infer/cpu.hpp>
#include <cuml/fil/treelite_importer.hpp>

namespace ML {
namespace fil {
namespace detail {
namespace device_initialization {

CUML_FIL_INITIALIZE_DEVICE(template, 0)
CUML_FIL_INITIALIZE_DEVICE(template, 1)
CUML_FIL_INITIALIZE_DEVICE(template, 2)
CUML_FIL_INITIALIZE_DEVICE(template, 3)
CUML_FIL_INITIALIZE_DEVICE(template, 4)
CUML_FIL_INITIALIZE_DEVICE(template, 5)
CUML_FIL_INITIALIZE_DEVICE(template, 6)
CUML_FIL_INITIALIZE_DEVICE(template, 7)
CUML_FIL_INITIALIZE_DEVICE(template, 8)
CUML_FIL_INITIALIZE_DEVICE(template, 9)
CUML_FIL_INITIALIZE_DEVICE(template, 10)
CUML_FIL_INITIALIZE_DEVICE(template, 11)

}  // namespace device_initialization

namespace inference {

CUML_FIL_INFER_ALL(template, raft_proto::device_type::cpu, 0)
CUML_FIL_INFER_ALL(template, raft_proto::device_type::cpu, 1)
CUML_FIL_INFER_ALL(template, raft_proto::device_type::cpu, 2)
CUML_FIL_INFER_ALL(template, raft_proto::device_type::cpu, 3)
CUML_FIL_INFER_ALL(template, raft_proto::device_type::cpu, 4)
CUML_FIL_INFER_ALL(template, raft_proto::device_type::cpu, 5)
CUML_FIL_INFER_ALL(template, raft_proto::device_type::cpu, 6)
CUML_FIL_INFER_ALL(template, raft_proto::device_type::cpu, 7)
CUML_FIL_INFER_ALL(template, raft_proto::device_type::cpu, 8)
CUML_FIL_INFER_ALL(template, raft_proto::device_type::cpu, 9)
CUML_FIL_INFER_ALL(template, raft_proto::device_type::cpu, 10)
CUML_FIL_INFER_ALL(template, raft_proto::device_type::cpu, 11)

}  // namespace inference
}  // namespace detail
}  // namespace fil
}  // namespace ML

namespace cuml4r {
namespace fil {
namespace {

__host__ int current_device() {
  int device = 0;
  CUDA_RT_CALL(cudaGetDevice(&device));
  return device;
}

}  // namespace

__host__ ML::fil::tree_layout tree_layout_from_storage_type(
  int const storage_type) {
  switch (storage_type) {
    case 1:
      return ML::fil::tree_layout::breadth_first;
    case 2:
      return ML::fil::tree_layout::depth_first;
    default:
      return ML::fil::tree_layout::depth_first;
  }
}

__host__ forest_uptr import_from_treelite(
  raft::handle_t const& handle, TreeliteHandle const& tl_handle,
  ML::fil::tree_layout const layout) {
  return std::make_unique<ML::fil::forest_model>(
    ML::fil::import_from_treelite_handle(
      /*tl_handle=*/tl_handle.handle(), /*layout=*/layout,
      /*align_bytes=*/128,
      /*use_double_precision=*/false,
      /*dev_type=*/raft_proto::device_type::gpu,
      /*device=*/current_device(),
      /*stream=*/handle.get_stream()));
}

__host__ void predict(raft::handle_t const& handle,
                      ML::fil::forest_model& forest, float* const output,
                      float* const input, std::size_t const num_rows,
                      ML::fil::infer_kind const infer_kind,
                      std::optional<ML::fil::index_type> const chunk_size) {
  raft_proto::handle_t fil_handle(handle);
  forest.predict(fil_handle, output, input, num_rows,
                 raft_proto::device_type::gpu, raft_proto::device_type::gpu,
                 infer_kind, chunk_size);
  fil_handle.synchronize();
}

}  // namespace fil
}  // namespace cuml4r

#endif
