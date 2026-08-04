#include <R_ext/Rdynload.h>
#include <Rinternals.h>

extern "C" {
SEXP _cuda_ml_nvforest_backend_versions();
SEXP _cuda_ml_nvforest_cpu_load_model(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                                      SEXP);
SEXP _cuda_ml_nvforest_cpu_model_info(SEXP);
SEXP _cuda_ml_nvforest_cpu_predict(SEXP, SEXP, SEXP, SEXP, SEXP);
SEXP _cuda_ml_nvforest_cpu_serialize(SEXP);
SEXP _cuda_ml_nvforest_cpu_unserialize(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                                       SEXP);
}

static const R_CallMethodDef CallEntries[] = {
    {"_cuda_ml_nvforest_backend_versions",
     reinterpret_cast<DL_FUNC>(&_cuda_ml_nvforest_backend_versions), 0},
    {"_cuda_ml_nvforest_cpu_load_model",
     reinterpret_cast<DL_FUNC>(&_cuda_ml_nvforest_cpu_load_model), 8},
    {"_cuda_ml_nvforest_cpu_model_info",
     reinterpret_cast<DL_FUNC>(&_cuda_ml_nvforest_cpu_model_info), 1},
    {"_cuda_ml_nvforest_cpu_predict",
     reinterpret_cast<DL_FUNC>(&_cuda_ml_nvforest_cpu_predict), 5},
    {"_cuda_ml_nvforest_cpu_serialize",
     reinterpret_cast<DL_FUNC>(&_cuda_ml_nvforest_cpu_serialize), 1},
    {"_cuda_ml_nvforest_cpu_unserialize",
     reinterpret_cast<DL_FUNC>(&_cuda_ml_nvforest_cpu_unserialize), 8},
    {nullptr, nullptr, 0}};

extern "C" void R_init_cuda_ml_nvforest(DllInfo *dll) {
  R_registerRoutines(dll, nullptr, CallEntries, nullptr, nullptr);
  R_useDynamicSymbols(dll, FALSE);
}
