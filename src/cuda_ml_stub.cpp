// The source-package stub never loads a backend. This registration-only
// translation unit lets R produce a conventional binary package without
// compiling CUDA or RAPIDS sources.
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>

extern "C" attribute_visible void R_init_cuda_ml(DllInfo* dll) {
  R_registerRoutines(dll, nullptr, nullptr, nullptr, nullptr);
  R_useDynamicSymbols(dll, FALSE);
}
