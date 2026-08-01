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
  unlink(
    file.path("tools", "backend", "src", x),
    recursive = TRUE,
    expand = TRUE
  )
}
