
#include <treelite/c_api.h>
#include <treelite/tree.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace treelite {

// Treelite's Python wheel exports its C API but hides this C++ function, which
// nvForest 26.06 calls from an inline public header. Keep the ownership boundary
// inside Treelite by delegating to the corresponding C API function.
std::unique_ptr<Model> ConcatenateModelObjects(
  std::vector<Model const*> const& models) {
  std::vector<TreeliteModelHandle> handles;
  handles.reserve(models.size());
  for (auto const* model : models) {
    handles.push_back(
      static_cast<TreeliteModelHandle>(const_cast<Model*>(model)));
  }

  TreeliteModelHandle result = nullptr;
  if (TreeliteConcatenateModelObjects(handles.data(), handles.size(),
                                      &result) != 0) {
    auto const* message = TreeliteGetLastError();
    throw std::runtime_error(
      message == nullptr ? "Treelite model concatenation failed." : message);
  }
  return std::unique_ptr<Model>(static_cast<Model*>(result));
}

}  // namespace treelite
