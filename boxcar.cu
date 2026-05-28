#include <cassert>

#include "boxcar.h"

namespace {

int readBit(int n, int bit_idx) {
  return (n & (1 << bit_idx)) >> bit_idx;
}

__global__ void gpu_boxcar_add_xy_z(int n, float* z, const float* x, const float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    z[i] = x[i] + y[i];
  }
}

__global__ void gpu_boxcar_scale(int n, float* z, const float* x, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    z[i] = x[i] * scale;
  }
}

void genBoxcarP2SumsGpu(float* gpu_p2_path_sums, const float* gpu_dd_sums_line, int n_freq,
                        int log2_max_p2, int n_zp) {
  int nbox_p2_max = 1 << log2_max_p2;
  int nbox_max = 2 * nbox_p2_max - 1;
  int n_freq_ext = n_freq + 2 * n_zp;
  int grid_size = (n_freq_ext + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;

  assert(n_zp >= nbox_max);

  for (int i_nbox = 0; i_nbox <= log2_max_p2; i_nbox++) {
    float* p2_row = &gpu_p2_path_sums[i_nbox * n_freq_ext];
    cudaMemsetAsync(&p2_row[0], 0, n_zp * sizeof(float));
    cudaMemsetAsync(&p2_row[n_freq + n_zp], 0, n_zp * sizeof(float));
    checkCuda("p2-cudaMemsetAsync");
  }

  float* p2_row = &gpu_p2_path_sums[0];
  cudaMemcpy(&p2_row[n_zp], gpu_dd_sums_line, n_freq * sizeof(float), cudaMemcpyDeviceToDevice);
  checkCuda("cudaMemcpy-p2=1");

  int stride = 1;
  for (int i_nbox = 1; i_nbox <= log2_max_p2; i_nbox++) {
    float* p2_row_new = &gpu_p2_path_sums[i_nbox * n_freq_ext];
    p2_row = &gpu_p2_path_sums[(i_nbox - 1) * n_freq_ext];
    int nbox_p2 = 1 << i_nbox;

    gpu_boxcar_add_xy_z<<<grid_size, CUDA_MAX_THREADS>>>(
        n_freq + n_zp + nbox_p2, &p2_row_new[stride], &p2_row[0], &p2_row[stride]);
    checkCuda("p2-gpu_boxcar_add_xy_z");
    stride = 2 * stride;
  }
}

void genBoxcarSumGpu(float* gpu_nbox_path_sum, float* gpu_p2_path_sums, float* gpu_work, int nbox,
                     int n_freq, int log2_max_p2, int n_zp) {
  assert(nbox >= 1);
  assert(nbox < (1 << (log2_max_p2 + 1)));

  int n_freq_ext = n_freq + 2 * n_zp;
  int grid_size = (n_freq_ext + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;

  cudaMemsetAsync(gpu_work, 0, 2 * n_freq_ext * sizeof(float));
  checkCuda("boxcar-cudaMemsetAsync");

  float* work_out_row = &gpu_work[0];
  const float* p2_row = &gpu_p2_path_sums[0];

  if (readBit(nbox, 0)) {
    cudaMemcpy(&work_out_row[0], &p2_row[0], n_freq_ext * sizeof(float), cudaMemcpyDeviceToDevice);
    checkCuda("cudaMemcpy-boxcar");
  }

  int work_out_idx = 0;
  int work_in_idx = 1;

  for (int i_nbox = 1; i_nbox <= log2_max_p2; i_nbox++) {
    int nbox_p2 = 1 << i_nbox;
    if (readBit(nbox, i_nbox) == 1) {
      work_in_idx = work_out_idx;
      work_out_idx = 1 - work_out_idx;

      work_out_row = &gpu_work[work_out_idx * n_freq_ext];
      const float* work_in_row = &gpu_work[work_in_idx * n_freq_ext];
      p2_row = &gpu_p2_path_sums[i_nbox * n_freq_ext];

      gpu_boxcar_add_xy_z<<<grid_size, CUDA_MAX_THREADS>>>(
          n_freq + n_zp, &work_out_row[nbox_p2], &work_in_row[0], &p2_row[nbox_p2]);
      checkCuda("boxcar-gpu_boxcar_add_xy_z");
    }
  }

  int shift = nbox / 2;
  float scale = 1.f / nbox;
  gpu_boxcar_scale<<<grid_size, CUDA_MAX_THREADS>>>(
      n_freq, gpu_nbox_path_sum, &work_out_row[n_zp + shift], scale);
  checkCuda("boxcar-gpu_boxcar_scale");
}

}  // namespace

BoxcarWorkspace::BoxcarWorkspace(int num_freq, const BoxcarConfig& config)
    : num_freq_(num_freq), config_(config) {
  int num_channels_ext = num_freq_ + 2 * config_.n_zp();
  cudaMalloc(&gpu_p2_path_sums_, num_channels_ext * config_.n_p2() * sizeof(float));
  checkCuda("p2_path_sums malloc");
  cudaMalloc(&gpu_boxcar_work_, 2 * num_channels_ext * sizeof(float));
  checkCuda("boxcar work malloc");
  cudaMalloc(&gpu_nbox_path_sum_, num_freq_ * sizeof(float));
  checkCuda("gpu_Nbox_path_sum malloc");
}

BoxcarWorkspace::~BoxcarWorkspace() {
  cudaFree(gpu_p2_path_sums_);
  cudaFree(gpu_boxcar_work_);
  cudaFree(gpu_nbox_path_sum_);
}

void BoxcarWorkspace::computeP2SumsGpu(const float* dd_sums_line, int log2_max_p2_runtime,
                                       int n_zp) {
  genBoxcarP2SumsGpu(gpu_p2_path_sums_, dd_sums_line, num_freq_, log2_max_p2_runtime, n_zp);
}

void BoxcarWorkspace::computeSumGpu(int nbox, int log2_max_p2_runtime, int n_zp) {
  genBoxcarSumGpu(gpu_nbox_path_sum_, gpu_p2_path_sums_, gpu_boxcar_work_, nbox, num_freq_,
                  log2_max_p2_runtime, n_zp);
}

const float* BoxcarWorkspace::filterLineGpu(const float* dd_sums_line, int nbox,
                                            int log2_max_p2_runtime) {
  computeP2SumsGpu(dd_sums_line, log2_max_p2_runtime, config_.n_zp());
  computeSumGpu(nbox, log2_max_p2_runtime, config_.n_zp());
  return gpu_nbox_path_sum_;
}
