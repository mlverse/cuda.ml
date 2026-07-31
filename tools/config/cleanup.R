for (x in c(
  "Makevars",
  "Makefile",
  "_deps",
  ".cmake-build",
  "CMakeCache.txt",
  "CMakeFiles",
  "cmake_install.cmake",
  "CMakeLists.txt",
  "symbols.rds",
  "*.o",
  "*.so"
)) {
  unlink(file.path("src", x), recursive = TRUE, expand = TRUE)
}

unlink(file.path("inst", "cuda-ml-backend.dcf"))
