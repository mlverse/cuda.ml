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
#define CUML4R_AS_HEX(x) CUML4R_CONCAT(0x, x)

#define CUML4R_TO_STRING_IMPL(x) #x
#define CUML4R_TO_STRING(x) CUML4R_TO_STRING_IMPL(x)

#ifdef __COUNTER__
#define CUML4R_ANONYMOUS_VARIABLE(x) \
  CUML4R_CONCAT(CUML4R_CONCAT(CUML4R_CONCAT(x, __LINE__), _), __COUNTER__)
#else
#define CUML4R_ANONYMOUS_VARIABLE(x) CUML4R_CONCAT(x, __LINE__)
#endif

#define CUML4R_ASSIGN_IF_PRESENT(X)                                          \
  template <typename T>                                                      \
  __host__ void set_##X(                                                     \
    T& t,                                                                    \
    typename std::remove_reference<decltype(T::X)>::type const x) noexcept { \
    t.X = x;                                                                 \
  }

#define CUML4R_NOOP_IF_ABSENT(X)          \
  template <typename T, typename... Args> \
  __host__ void set_##X(T&, Args...) noexcept {}

// Workaround for libcuml minor version such as '08' being "misinterpreted" as
// invalid octal numbers
// (the low-order 16 bits should be more than sufficient for storing the minor
//  version number)
#define CUML4R_LIBCUML_VERSION(version_major, version_minor) \
  ((CUML4R_AS_HEX(version_major) << 16) | CUML4R_AS_HEX(version_minor))
