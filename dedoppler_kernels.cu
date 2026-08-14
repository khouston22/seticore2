#include "dedoppler_kernels.h"

// Dedoppler: drift path SNR kernels

// Update per-frequency top path SNR for one drift/nbox step
__global__ void findTopPathSNRsGPU(const float* path_sums_line, int num_timesteps, int rounded_num_timesteps,
                                    int num_freqs, int path_offset, int drift_block, float* mu,
                                    float* sigma_scale, int nbox, float* top_path_snrs,
                                    int* top_drift_blocks, int* top_path_offsets,
                                    int* top_path_nbox) {
  int freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (freq < 0 || freq >= num_freqs) {
    return;
  }

  float path_scale = 1.f / num_timesteps;

  if (drift_block >= 0) {
    int last_freq = num_freqs - 1 - ((rounded_num_timesteps - 1) * drift_block + path_offset) - nbox;
    if (freq > last_freq) {
      return;
    }
  } else {
    int first_freq = -((rounded_num_timesteps - 1) * drift_block + path_offset) + nbox;
    if (freq < first_freq) {
      return;
    }
  }

  float path_snr = (path_sums_line[freq] * path_scale - mu[freq]) * sigma_scale[freq];
  if (path_snr > top_path_snrs[freq]) {
    top_path_snrs[freq] = path_snr;
    top_drift_blocks[freq] = drift_block;
    top_path_offsets[freq] = path_offset;
    top_path_nbox[freq] = nbox;
  }
}

// Dedoppler: column sum kernels

// Sum spectrogram columns on GPU with scale factor
__global__ void sumColumns(const float* input, float* sums, int num_timesteps, int num_freqs,
                           float scale) {
  int freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (freq < 0 || freq >= num_freqs) {
    return;
  }
  sums[freq] = 0.0f;
  for (int i = freq; i < num_timesteps * num_freqs; i += num_freqs) {
    sums[freq] += input[i];
  }
  sums[freq] *= scale;
}

// Sum spectrogram columns on CPU (reference path)
void sumColumnsCpu(const float* input, float* sums, int num_timesteps, int n_freq) {
  for (int time = 0; time < num_timesteps; time++) {
    if (time == 0) {
      for (int freq = 0; freq < n_freq; freq++) {
        sums[freq] = input[freq];
      }
    } else {
      int in_ofs = time * n_freq;
      for (int freq = 0; freq < n_freq; freq++) {
        sums[freq] += input[in_ofs++];
      }
    }
  }
}

// Launch sumColumns kernel
void launchSumColumns(const float* input, float* sums, int num_timesteps, int num_freqs,
                      float scale) {
  int grid_size = (num_freqs + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;
  sumColumns<<<grid_size, CUDA_MAX_THREADS>>>(input, sums, num_timesteps, num_freqs, scale);
  checkCuda("sumColumns");
}

// Launch findTopPathSNRsGPU kernel
void launchFindTopPathSNRs(const float* path_sums_line, int num_timesteps, int rounded_num_timesteps,
                           int num_freqs, int path_offset, int drift_block, float* mu, float* sigma_scale,
                           int nbox, float* top_path_snrs, int* top_drift_blocks,
                           int* top_path_offsets, int* top_path_nbox) {
  int grid_size = (num_freqs + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;
  findTopPathSNRsGPU<<<grid_size, CUDA_MAX_THREADS>>>(
      path_sums_line, num_timesteps, rounded_num_timesteps, num_freqs, path_offset, drift_block, mu, sigma_scale, nbox,
      top_path_snrs, top_drift_blocks, top_path_offsets, top_path_nbox);
  checkCuda("findTopPathSNRsGPU");
}
