#pragma once

#if HAS_CUML

#include <raft/handle.hpp>

#include <memory>

namespace cuml4r {

template <typename M>
struct ModelCtx {
  std::unique_ptr<M> const model_;
  std::unique_ptr<raft::handle_t> const handle_;

  ModelCtx(std::unique_ptr<M> model,
           std::unique_ptr<raft::handle_t> handle) noexcept
    : model_(std::move(model)), handle_(std::move(handle)) {}
};

}  // namespace cuml4r

#else

#include "warn_cuml_missing.h"

#endif
