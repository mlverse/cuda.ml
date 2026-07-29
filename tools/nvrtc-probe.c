#include <nvrtc.h>

#include <stdio.h>
#include <stdlib.h>

static void print_program_log(nvrtcProgram program) {
  size_t size = 0;
  nvrtcResult result = nvrtcGetProgramLogSize(program, &size);
  if (result != NVRTC_SUCCESS) {
    fprintf(
        stderr,
        "nvrtcGetProgramLogSize failed: %s\n",
        nvrtcGetErrorString(result));
    return;
  }

  char *log = malloc(size);
  if (log == NULL) {
    fputs("Unable to allocate the NVRTC program log.\n", stderr);
    return;
  }

  result = nvrtcGetProgramLog(program, log);
  if (result == NVRTC_SUCCESS) {
    fputs(log, stderr);
  } else {
    fprintf(
        stderr,
        "nvrtcGetProgramLog failed: %s\n",
        nvrtcGetErrorString(result));
  }
  free(log);
}

int main(void) {
  static const char source[] =
      "extern \"C\" __global__ void probe(float *x) {"
      "  x[threadIdx.x] += 1.0f;"
      "}";
  static const char *options[] = {
      "--gpu-architecture=compute_75",
  };
  nvrtcProgram program;

  nvrtcResult result =
      nvrtcCreateProgram(&program, source, "nvrtc-probe.cu", 0, NULL, NULL);
  if (result != NVRTC_SUCCESS) {
    fprintf(
        stderr,
        "nvrtcCreateProgram failed: %s\n",
        nvrtcGetErrorString(result));
    return EXIT_FAILURE;
  }

  result = nvrtcCompileProgram(program, 1, options);
  if (result != NVRTC_SUCCESS) {
    fprintf(
        stderr,
        "nvrtcCompileProgram failed: %s\n",
        nvrtcGetErrorString(result));
    print_program_log(program);
    nvrtcDestroyProgram(&program);
    return EXIT_FAILURE;
  }

  result = nvrtcDestroyProgram(&program);
  if (result != NVRTC_SUCCESS) {
    fprintf(
        stderr,
        "nvrtcDestroyProgram failed: %s\n",
        nvrtcGetErrorString(result));
    return EXIT_FAILURE;
  }

  puts("NVRTC compiled compute_75 without a GPU.");
  return EXIT_SUCCESS;
}
