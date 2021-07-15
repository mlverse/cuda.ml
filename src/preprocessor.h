#pragma once

#ifndef __has_cpp_attribute
#define __has_cpp_attribute(x) 0
#endif

#if __has_cpp_attribute(nodiscard)
#define CUML4R_NODISCARD [[nodiscard]]
#else
#define CUML4R_NODISCARD
#endif

// NOTE: the idea for the following is borrowed from
// https://github.com/facebook/folly/blob/7a18d1823185495cae6676258ee64afd7e36c84c/folly/Preprocessor.h#L88-L105
#define CUML4R_CONCAT_IMPL(a, b) a##b
#define CUML4R_CONCAT(a, b) CUML4R_CONCAT_IMPL(a, b)

#ifdef __COUNTER__
#define CUML4R_ANONYMOUS_VARIABLE(x) \
  CUML4R_CONCAT(CUML4R_CONCAT(CUML4R_CONCAT(x, __LINE__), _), __COUNTER__)
#else
#define CUML4R_ANONYMOUS_VARIABLE(x) CUML4R_CONCAT(x, __LINE__)
#endif
