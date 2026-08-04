cmake_minimum_required(VERSION 3.21.1 FATAL_ERROR)

project(cuda.ml.nvforest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
  message(FATAL_ERROR
    "The pinned nvForest 26.06 binary requires GCC 14 or newer."
  )
endif()

foreach(required R_INCLUDE_DIR RCPP_INCLUDE_DIR RAPIDS_INCLUDE_DIR RAPIDS_LIB_DIR)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required.")
  endif()
endforeach()

set(TREELITE_STATIC_LIBRARY "${RAPIDS_LIB_DIR}/libtreelite_static.a")
if(NOT EXISTS "${TREELITE_STATIC_LIBRARY}")
  message(FATAL_ERROR "libtreelite_static.a was not found in ${RAPIDS_LIB_DIR}.")
endif()

set(NVFOREST_IMPLEMENTATION "${CMAKE_CURRENT_SOURCE_DIR}/../backend-src/nvforest.cu")
set_source_files_properties(
  "${NVFOREST_IMPLEMENTATION}"
  PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++ -fvisibility=hidden"
)

add_library(
  cuda.ml.nvforest
  SHARED
  "${CMAKE_CURRENT_SOURCE_DIR}/../backend-src/nvforest.cpp"
  "${NVFOREST_IMPLEMENTATION}"
  backend_versions.cpp
  cpu_infer.cpp
  exports.cpp
  native_symbols.cpp
)

set_target_properties(
  cuda.ml.nvforest
  PROPERTIES
  PREFIX ""
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN"
  INSTALL_RPATH_USE_LINK_PATH FALSE
)

target_compile_definitions(
  cuda.ml.nvforest
  PRIVATE
  NDEBUG
  CUDA_ML_NVFOREST_CPU_ONLY
  NVFOREST_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_OFF
  RMM_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_OFF
)

target_include_directories(
  cuda.ml.nvforest
  SYSTEM PRIVATE
  ${R_INCLUDE_DIR}
  ${RCPP_INCLUDE_DIR}
  ${RAPIDS_INCLUDE_DIR}/rapids
  ${RAPIDS_INCLUDE_DIR}
)
target_include_directories(
  cuda.ml.nvforest
  PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../backend-src"
)

target_link_options(
  cuda.ml.nvforest
  PRIVATE
  "LINKER:--exclude-libs,libtreelite_static.a"
  "LINKER:-Bsymbolic-functions"
)
target_link_libraries(
  cuda.ml.nvforest
  PRIVATE
  ${TREELITE_STATIC_LIBRARY}
)
