#pragma once

#ifndef CUML4R_TREELITE_C_API_MISSING

#include "treelite_utils.cuh"

#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/forest_model.hpp>
#include <cuml/fil/infer_kind.hpp>
#include <cuml/fil/tree_layout.hpp>

#include <cstddef>
#include <memory>
#include <optional>

namespace cuml4r {
namespace fil {

using forest_uptr = std::unique_ptr<ML::fil::forest_model>;

ML::fil::tree_layout tree_layout_from_storage_type(int storage_type);

forest_uptr import_from_treelite(
  raft::handle_t const& handle, TreeliteHandle const& tl_handle,
  ML::fil::tree_layout layout = ML::fil::tree_layout::depth_first);

void predict(raft::handle_t const& handle, ML::fil::forest_model& forest,
             float* output, float* input, std::size_t num_rows,
             ML::fil::infer_kind infer_kind = ML::fil::infer_kind::default_kind,
             std::optional<ML::fil::index_type> chunk_size =
               std::optional<ML::fil::index_type>{4});

}  // namespace fil
}  // namespace cuml4r

#endif
