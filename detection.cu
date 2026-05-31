#include "detection.h"

// Detection: subband normalization kernels

// Linearly interpolate subband mean/std to full frequency axis
__global__ void gpu_subband_interpolate(float* x, int n_freq, const float* x_subband,
                                        int n_subband) {
  int i_freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_freq < 0 || i_freq >= n_freq) {
    return;
  }

  int nf_subband = n_freq / n_subband;
  int i_subband = i_freq / nf_subband;
  int df = i_freq - nf_subband * i_subband - nf_subband / 2;
  float scale;

  if (df >= 0) {
    if (i_subband < n_subband - 1) {
      scale = (x_subband[i_subband + 1] - x_subband[i_subband]) / nf_subband;
    } else {
      scale = 0.f;
    }
  } else {
    if (i_subband > 0) {
      scale = (x_subband[i_subband] - x_subband[i_subband - 1]) / nf_subband;
    } else {
      scale = 0.f;
    }
  }
  x[i_freq] = x_subband[i_subband] + df * scale;
}

// Divide spectrogram row by per-frequency mean (equalize)
__global__ void gpu_local_mean_scale(float* x, const float* mu, int n_freq) {
  int i_freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_freq < 0 || i_freq >= n_freq) {
    return;
  }
  x[i_freq] = x[i_freq] / mu[i_freq];
}

// Per-frequency SNR scale factor from std and nbox gain
__global__ void gpu_compute_sigma_scale(float* sigma_scale, const float* sigma, float nbox_gain,
                                        int n_freq) {
  int i_freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_freq < 0 || i_freq >= n_freq) {
    return;
  }
  sigma_scale[i_freq] = nbox_gain / sigma[i_freq];
}

// Detection: stamp extraction kernel

// Copy drift stamp submatrix with centered boxcar averaging
__global__ void gpu_copy_submatrix_boxcar(float* dst_matrix, const float* src_matrix,
                                          int src_n_rows, int src_n_freq, int start_freq,
                                          int n_freq_to_copy, int start_row, int n_rows_to_copy,
                                          int nbox) {
  int i_freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (i_freq < start_freq || i_freq >= start_freq + n_freq_to_copy) {
    return;
  }

  float scale = 1.f / nbox;
  for (int i_row = 0; i_row < n_rows_to_copy; i_row++) {
    float temp_sum = 0.f;
    for (int i_box = 0; i_box < nbox; i_box++) {
      temp_sum += src_matrix[i_row * src_n_freq + i_freq + i_box - nbox / 2];
    }
    dst_matrix[i_row * n_freq_to_copy + i_freq - start_freq] = temp_sum * scale;
  }
}

// SubbandNormalizer: GPU launch wrappers

// Launch subband-to-frequency interpolation
void SubbandNormalizer::interpolateToFreqGpu(float* gpu_out, int n_freq, const float* gpu_subband,
                                             int n_subband) {
  int grid_size = (n_freq + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;
  gpu_subband_interpolate<<<grid_size, CUDA_MAX_THREADS>>>(gpu_out, n_freq, gpu_subband, n_subband);
  checkCuda("gpu_subband_interpolate");
}

// Launch spectrogram row equalization
void SubbandNormalizer::equalizeSpectrogramRowGpu(float* d_sg_row, const float* gpu_mu,
                                                  int n_freq) {
  int grid_size = (n_freq + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;
  gpu_local_mean_scale<<<grid_size, CUDA_MAX_THREADS>>>(d_sg_row, gpu_mu, n_freq);
  checkCuda("gpu_local_mean_scale");
}

// Launch sigma scale computation
void SubbandNormalizer::computeSigmaScaleGpu(float* gpu_sigma_scale, const float* gpu_std,
                                             float nbox_gain, int n_freq) {
  int grid_size = (n_freq + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;
  gpu_compute_sigma_scale<<<grid_size, CUDA_MAX_THREADS>>>(gpu_sigma_scale, gpu_std, nbox_gain,
                                                            n_freq);
  checkCuda("gpu_compute_sigma_scale");
}

// StampAnalyzer: GPU stamp extraction

// Launch boxcar-averaged stamp submatrix copy
void StampAnalyzer::extractStampGpu(float* dst, const float* src, int src_n_rows, int src_n_freq,
                                    int start_freq, int n_freq_to_copy, int start_row,
                                    int n_rows_to_copy, int nbox) {
  int grid_size = (src_n_freq + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;
  gpu_copy_submatrix_boxcar<<<grid_size, CUDA_MAX_THREADS>>>(
      dst, src, src_n_rows, src_n_freq, start_freq, n_freq_to_copy, start_row, n_rows_to_copy, nbox);
  checkCuda("Stamp gpu_copy_submatrix");
}
