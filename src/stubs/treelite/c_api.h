#pragma once

#define CUML4R_TREELITE_C_API_MISSING

#pragma message( \
"Treelite C API header is missing. Forest Inference Library (FIL) functionalities from `cuml4r` will be disabled. Please consider installing `treelite` C libraries and re-installing `cuml4r` if you want FIL enabled.")
