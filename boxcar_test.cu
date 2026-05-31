
#include "boxcar.h"
#include <cstring>
#include <fmt/core.h>

using namespace std;

#define TEST_GPU 1

int main() {
  int n_freq = 1 << 12;
  BoxcarConfig config;
  config.log2_max_p2 = 4;
  int nbox_p2_max = config.nbox_p2_max();
  int nbox_max = config.nbox_max();
  int n_zp = config.n_zp();
  int n_freq_ext = n_freq + 2 * n_zp;
  int n_p2 = config.n_p2();

  float* dd_sums_line;
  float* p2_path_sums;
  float* work;
  float* nbox_path_sum;

#if TEST_GPU
  fmt::print("Boxcar GPU Test\n");
  float *gpu_dd_sums_line;
  cudaMalloc(&gpu_dd_sums_line, n_freq * sizeof(float));
  cudaMallocHost(&dd_sums_line, n_freq * sizeof(float));
  checkCuda("DD_sums_line malloc");
  BoxcarWorkspace boxcar(n_freq, config);
#else
  fmt::print("Boxcar CPU Test\n");
  dd_sums_line = static_cast<float*>(malloc(n_freq * sizeof(float)));
  p2_path_sums = static_cast<float*>(malloc(n_freq_ext * n_p2 * sizeof(float)));
  work = static_cast<float*>(malloc(2 * n_freq_ext * sizeof(float)));
  nbox_path_sum = static_cast<float*>(malloc(n_freq * sizeof(float)));
  BoxcarWorkspace boxcar(n_freq, config);
#endif

  fmt::print("n_freq={}, log2_max_p2={}, Nbox_p2_max={}, Nbox_max={}, n_zp={}\n", n_freq,
             config.log2_max_p2, nbox_p2_max, nbox_max, n_zp);

  memset(dd_sums_line, 0, n_freq * sizeof(float));

  int sig_start = n_freq / 2;
  float sig_value = 1.0f;

#if 1
  int sig_width = 1;
  for (int i_freq = sig_start; i_freq < sig_start + sig_width; i_freq++) {
    dd_sums_line[i_freq] = sig_value;
  }
#else
  int sig_width = 5;
  for (int i_freq = sig_start; i_freq < sig_start + sig_width; i_freq++) {
    dd_sums_line[i_freq] = sig_value + i_freq - sig_start;
  }
#endif

#if TEST_GPU
  cudaMallocHost(&p2_path_sums, n_freq_ext * n_p2 * sizeof(float));
  cudaMallocHost(&work, 2 * n_freq_ext * sizeof(float));
  cudaMallocHost(&nbox_path_sum, n_freq * sizeof(float));
  checkCuda("host buffers malloc");

  cudaMemcpy(gpu_dd_sums_line, dd_sums_line, n_freq * sizeof(float), cudaMemcpyHostToDevice);
  checkCuda("cudaMemcpy-DD_sums_line");

  boxcar.computeP2SumsGpu(gpu_dd_sums_line, config.log2_max_p2, n_zp);
  cudaMemcpy(p2_path_sums, boxcar.gpuP2PathSums(), n_freq_ext * n_p2 * sizeof(float),
             cudaMemcpyDeviceToHost);
  checkCuda("cudaMemcpy-gen_boxcar_p2_sums_gpu");
  cudaDeviceSynchronize();
#else
  boxcar.computeP2SumsCpu(dd_sums_line, config.log2_max_p2, p2_path_sums, n_zp);
#endif

  int print_ofs = -20;
  int print_n_pts = 60;

  for (int i_nbox = 0; i_nbox <= config.log2_max_p2; i_nbox++) {
    int start_idx = n_freq_ext * i_nbox + sig_start + n_zp;
    int nbox_p2 = 1 << i_nbox;
    fmt::print("\nNbox_p2={}, sig_start={} {}\n", nbox_p2, sig_start, n_freq - sig_start);
    int n_pts = min(print_n_pts, n_freq_ext - (sig_start - print_ofs));
    BoxcarWorkspace::printNboxSegment(&p2_path_sums[start_idx], n_pts, print_ofs, 1.f);
  }

  for (int nbox = 1; nbox <= nbox_max; nbox++) {
#if TEST_GPU
    boxcar.computeP2SumsGpu(gpu_dd_sums_line, config.log2_max_p2, n_zp);
    boxcar.computeSumGpu(nbox, config.log2_max_p2, n_zp);
    cudaMemcpy(nbox_path_sum, boxcar.gpuNboxPathSum(), n_freq * sizeof(float),
               cudaMemcpyDeviceToHost);
    checkCuda("cudaMemcpy-gen_boxcar_sum_gpu");
    cudaDeviceSynchronize();
#else
    boxcar.computeSumCpu(p2_path_sums, work, nbox, config.log2_max_p2, n_zp, nbox_path_sum);
#endif

    int start_idx = sig_start;
    fmt::print("\nNbox={}, sig_start={} {}\n", nbox, sig_start, n_freq - sig_start);
    int n_pts = min(print_n_pts, n_freq - start_idx);
    BoxcarWorkspace::printNboxSegment(&nbox_path_sum[start_idx], n_pts, print_ofs,
                                      static_cast<float>(nbox));
  }

#if TEST_GPU
  cudaFree(gpu_dd_sums_line);
  cudaFreeHost(dd_sums_line);
  cudaFreeHost(p2_path_sums);
  cudaFreeHost(work);
  cudaFreeHost(nbox_path_sum);
#else
  free(dd_sums_line);
  free(p2_path_sums);
  free(work);
  free(nbox_path_sum);
#endif

  return 0;
}
