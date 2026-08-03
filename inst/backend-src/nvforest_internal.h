#pragma once

#include "treelite_utils.cuh"

#include <Rinternals.h>

namespace cuml4r {

enum class NvForestDevice : int { CPU = 0, GPU = 1 };
enum class NvForestLayout : int {
  DEPTH_FIRST = 0,
  BREADTH_FIRST = 1,
  LAYERED = 2
};
enum class NvForestPrecision : int { NATIVE = -1, SINGLE = 0, DOUBLE = 1 };

struct NvForestOptions {
  NvForestDevice device;
  int device_id;
  NvForestLayout layout;
  NvForestPrecision precision;
  int default_chunk_size;
  int align_bytes;
};

SEXP nvforest_from_treelite(TreeliteHandle&& treelite,
                            NvForestOptions const& options,
                            bool averaged_vector_leaf_probabilities);

}  // namespace cuml4r
