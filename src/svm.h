#pragma once

#if HAS_CUML

#include <memory>

namespace raft {

class handle_t;

}  // namespace raft

namespace cuml4r {

template <typename M, typename H = raft::handle_t>
struct ModelCtx {
  std::unique_ptr<M> const model_;
  std::unique_ptr<H> const handle_;

  ModelCtx(std::unique_ptr<M> model, std::unique_ptr<H> handle) noexcept
    : model_(std::move(model)), handle_(std::move(handle)) {}
};

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
