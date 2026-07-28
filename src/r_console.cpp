#include <R_ext/Print.h>

#include <cstdarg>
#include <cstdio>

extern "C" {

// GNU ld redirects RAPIDS/Thrust output to these R console shims.
FILE* __wrap_stderr = nullptr;

int __wrap_printf(char const* format, ...)
{
  std::va_list args;
  va_start(args, format);
  Rvprintf(format, args);
  va_end(args);
  return 0;
}

int __wrap_fprintf(FILE*, char const* format, ...)
{
  std::va_list args;
  va_start(args, format);
  REvprintf(format, args);
  va_end(args);
  return 0;
}

}
