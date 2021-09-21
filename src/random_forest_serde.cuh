#pragma once

#ifndef CUML4R_TREELITE_C_API_MISSING

#include <treelite/tree.h>

#include <Rcpp.h>

#include <memory>

namespace cuml4r {
namespace detail {

// Non-POD PyBufFrame states that must be copied onto the heap and kept alive
// until after the resulting Treelite model is destroyed when reconstructing
// a rand_forest model from a serialized model state.
class PyBufFrameContent {
 public:
  __host__ PyBufFrameContent(std::string const& format,
                             std::vector<Rbyte> const& buf)
    : impl_(std::make_unique<Impl>(format, buf)) {}
  __host__ PyBufFrameContent(PyBufFrameContent&& o) noexcept
    : impl_(std::move(o.impl_)) {}
  __host__ auto const* operator-> () const noexcept { return impl_.get(); }

 private:
  struct Impl {
    std::string const format_;
    std::vector<Rbyte> const buf_;
    __host__ Impl(std::string const& format, std::vector<Rbyte> const& buf)
      : format_(format), buf_(buf) {}
  };
  std::unique_ptr<Impl> impl_;
};

Rcpp::List getState(treelite::PyBufferFrame const&);

Rcpp::List getState(treelite::Model const&);

void setState(treelite::PyBufferFrame&, std::vector<PyBufFrameContent>&,
              Rcpp::List const&);

void setState(std::unique_ptr<treelite::Model>&,
              std::vector<PyBufFrameContent>&, Rcpp::List const&);

}  // namespace detail
}  // namespace cuml4r

#endif
