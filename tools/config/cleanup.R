for (x in c("Makevars", "Makefile", "CMakeCache.txt", "CMakeFiles", "cmake_install.cmake", "CMakeLists.txt", "*.o", "*.so")) {
  unlink(file.path("src", x), recursive = TRUE, expand = TRUE)
}
