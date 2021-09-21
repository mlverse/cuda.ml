#include "random_forest_serde.cuh"

#ifndef CUML4R_TREELITE_C_API_MISSING

#include <algorithm>

namespace cuml4r {
namespace detail {

constexpr auto kPyBufFrameContent = "buf";
constexpr auto kPyBufFrameFormat = "format";
constexpr auto kPyBufFrameItemSize = "itemsize";
constexpr auto kPyBufFrameNumItems = "nitem";

__host__ Rcpp::List getState(treelite::PyBufferFrame const& buf) {
  Rcpp::List state;

  {
    auto const sz = buf.itemsize * buf.nitem;
    std::vector<Rbyte> content(sz);
    std::copy(reinterpret_cast<unsigned char*>(buf.buf),
              reinterpret_cast<unsigned char*>(buf.buf) + sz, content.begin());
    state[kPyBufFrameContent] = std::move(content);
  }
  state[kPyBufFrameFormat] = std::string(buf.format);
  state[kPyBufFrameItemSize] = buf.itemsize;
  state[kPyBufFrameNumItems] = buf.nitem;

  return state;
}

__host__ Rcpp::List getState(treelite::Model const& model) {
  Rcpp::List state;

  auto const buffers = const_cast<treelite::Model&>(model).GetPyBuffer();
  for (auto const& buffer : buffers) {
    state.push_back(getState(buffer));
  }

  return state;
}

__host__ void setState(treelite::PyBufferFrame& frame,
                       std::vector<PyBufFrameContent>& frames_content,
                       Rcpp::List const& state) {
  PyBufFrameContent frame_content(
    /*format=*/Rcpp::as<std::string>(state[kPyBufFrameFormat]),
    /*buf=*/Rcpp::as<std::vector<Rbyte>>(state[kPyBufFrameContent]));

  frame.itemsize = state[kPyBufFrameItemSize];
  frame.nitem = state[kPyBufFrameNumItems];
  frame.format = const_cast<char*>(frame_content->format_.c_str());
  frame.buf = const_cast<Rbyte*>(frame_content->buf_.data());

  frames_content.emplace_back(std::move(frame_content));
}

void setState(std::unique_ptr<treelite::Model>& model,
              std::vector<PyBufFrameContent>& frames_content,
              Rcpp::List const& state) {
  frames_content.reserve(state.size());
  std::vector<treelite::PyBufferFrame> frames(state.size());

  for (size_t i = 0; i < state.size(); ++i) {
    setState(/*buf=*/frames[i], frames_content, /*state=*/state[i]);
  }

  model = treelite::Model::CreateFromPyBuffer(frames);
}

}  // namespace detail
}  // namespace cuml4r

#endif
