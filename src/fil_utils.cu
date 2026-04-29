#include "fil_utils.h"

namespace cuml4r {
namespace fil {

__host__ forest_uptr make_forest(raft::handle_t const& handle,
                                 ML::fil::forest* const forest) {
  return forest_uptr(forest, [&handle](auto* const f) {
    if (f != nullptr) {
      ML::fil::free(handle, f);
    }
  });
}

__host__ forest_uptr make_forest(raft::handle_t const& handle,
                                 std::function<ML::fil::forest*()> src) {
  return make_forest(handle, src());
}

}  // namespace fil
}  // namespace cuml4r
