// [[Rcpp::export(".has_libcuml")]]
bool has_libcuml() {
#ifdef HAS_CUML

  return true;

#else

  return false;

#endif
}
