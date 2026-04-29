#pragma once

#include <cuml/fil/fil.h>

#include <functional>
#include <memory>

namespace cuml4r {
namespace fil {

using forest_uptr =
  std::unique_ptr<ML::fil::forest, std::function<void(ML::fil::forest* const)>>;

/*
 * RAII wrapper for a `ML::fil::forest` pointer (a.k.a `ML::fil::forest_t`)
 *
 * NOTE: the resulting RAII wrapper does *not* take ownship of `handle`, and
 * assumes `handle` will be destroyed *after* the FIL forest object itself is
 * destroyed.
 */
forest_uptr make_forest(raft::handle_t const& handle,
                        ML::fil::forest* const forest);

forest_uptr make_forest(raft::handle_t const& handle,
                        std::function<ML::fil::forest*()> src);

}  // namespace fil
}  // namespace cuml4r
