nvforest_cpu_pointer <- function(xptr) {
  identical(attr(xptr, "cuda_ml_nvforest_backend", exact = TRUE), "cpu")
}

nvforest_tag_cpu_pointer <- function(xptr) {
  attr(xptr, "cuda_ml_nvforest_backend") <- "cpu"
  xptr
}

nvforest_cpu_call <- function(symbol, ...) {
  do.call(
    .Call,
    c(list(cuda_ml_nvforest_cpu_symbol(symbol)), list(...))
  )
}

nvforest_use_slim_cpu_backend <- function() {
  info <- cuda_ml_backend_info()
  info$nvforest_cpu_runtime_installed || !info$runtime_installed
}

nvforest_native_load_model <- function(
  filename,
  model_type,
  device,
  device_id,
  layout,
  precision,
  default_chunk_size,
  align_bytes
) {
  if (identical(device, 0L) && nvforest_use_slim_cpu_backend()) {
    return(nvforest_tag_cpu_pointer(nvforest_cpu_call(
      "_cuda_ml_nvforest_cpu_load_model",
      filename,
      model_type,
      device,
      device_id,
      layout,
      precision,
      default_chunk_size,
      align_bytes
    )))
  }
  .nvforest_load_model(
    filename,
    model_type,
    device,
    device_id,
    layout,
    precision,
    default_chunk_size,
    align_bytes
  )
}

nvforest_native_model_info <- function(xptr) {
  if (nvforest_cpu_pointer(xptr)) {
    return(nvforest_cpu_call("_cuda_ml_nvforest_cpu_model_info", xptr))
  }
  .nvforest_model_info(xptr)
}

nvforest_native_predict <- function(
  model,
  input,
  prediction_type,
  threshold,
  chunk_size
) {
  if (nvforest_cpu_pointer(model)) {
    return(nvforest_cpu_call(
      "_cuda_ml_nvforest_cpu_predict",
      model,
      input,
      prediction_type,
      threshold,
      chunk_size
    ))
  }
  .nvforest_predict(model, input, prediction_type, threshold, chunk_size)
}

nvforest_native_serialize <- function(xptr) {
  if (nvforest_cpu_pointer(xptr)) {
    return(nvforest_cpu_call("_cuda_ml_nvforest_cpu_serialize", xptr))
  }
  .nvforest_serialize(xptr)
}

nvforest_native_unserialize <- function(
  bytes,
  device,
  device_id,
  layout,
  precision,
  default_chunk_size,
  align_bytes,
  averaged_vector_leaf_probabilities
) {
  if (identical(device, 0L) && nvforest_use_slim_cpu_backend()) {
    return(nvforest_tag_cpu_pointer(nvforest_cpu_call(
      "_cuda_ml_nvforest_cpu_unserialize",
      bytes,
      device,
      device_id,
      layout,
      precision,
      default_chunk_size,
      align_bytes,
      averaged_vector_leaf_probabilities
    )))
  }
  .nvforest_unserialize(
    bytes,
    device,
    device_id,
    layout,
    precision,
    default_chunk_size,
    align_bytes,
    averaged_vector_leaf_probabilities
  )
}
