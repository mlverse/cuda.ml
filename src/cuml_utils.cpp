// [[Rcpp::export(".has_cuml")]]
bool has_cuml() {
#ifdef HAS_CUML

  return true;

#else

  return false;

#endif
}
