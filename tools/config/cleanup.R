for (x in c("Makevars", "Makefile", ".cmake-build", "CMakeCache.txt", "CMakeFiles", "cmake_install.cmake", "CMakeLists.txt", "symbols.rds", "*.o", "*.so")) {
  unlink(file.path("src", x), recursive = TRUE, expand = TRUE)
}
