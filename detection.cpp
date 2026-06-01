#include <algorithm>
#include <cmath>
#include <fmt/core.h>

#include "detection.h"

// StatsUtil: mean, min/max, DC spike replacement

// Sample mean and std dev (double accumulators)
void StatsUtil::meanStdDev(const float* x, int n, float* mean, float* std_dev) {
  double sum_x2 = 0.;
  double sum_x = 0.;

  for (int i = 0; i < n; i++) {
    float temp = x[i];
    sum_x += temp;
    sum_x2 += temp * temp;
  }
  *mean = static_cast<float>(sum_x / n);
  *std_dev = sqrt((sum_x2 - n * (*mean) * (*mean)) / (n - 1));
}

// Sample mean and std dev (float accumulators)
void StatsUtil::meanStdDev2(const float* x, int n, float* mean, float* std_dev) {
  float sum_x2 = 0.f;
  float sum_x = 0.f;

  for (int i = 0; i < n; i++) {
    float temp = x[i];
    sum_x += temp;
    sum_x2 += temp * temp;
  }
  *mean = sum_x / n;
  *std_dev = sqrt((sum_x2 - n * (*mean) * (*mean)) / (n - 1));
}

// Maximum value in array
float StatsUtil::max(const float* x, int n) {
  float x_max = x[0];
  for (int i = 1; i < n; i++) {
    x_max = std::max(x_max, x[i]);
  }
  return x_max;
}

// Minimum value in array
float StatsUtil::min(const float* x, int n) {
  float x_min = x[0];
  for (int i = 1; i < n; i++) {
    x_min = std::min(x_min, x[i]);
  }
  return x_min;
}

// Replace DC spike bins with mean of adjacent samples
void StatsUtil::replaceDcSpike(float* x, int dc_ofs, int mean_pts) {
  float adj_mean = 0.f;
  for (int i_ofs = -dc_ofs - mean_pts; i_ofs < -dc_ofs; i_ofs++) {
    adj_mean += x[i_ofs];
  }
  for (int i_ofs = dc_ofs + 1; i_ofs <= dc_ofs + mean_pts; i_ofs++) {
    adj_mean += x[i_ofs];
  }
  adj_mean /= (2 * mean_pts);

  for (int i_ofs = -dc_ofs; i_ofs <= dc_ofs; i_ofs++) {
    x[i_ofs] = adj_mean;
  }
}

// StatsUtil: debug array printers

// Debug print array centered at index 0 (negative offsets)
void StatsUtil::printXLr(float* x, int max_ofs, float scale) {
  for (int i_ofs = -max_ofs; i_ofs < max_ofs; i_ofs++) {
    if (i_ofs % 10 == 0) {
      fmt::print("\n{:6d}   ", i_ofs);
    }
    fmt::print("{:8.0f} ", x[i_ofs] * scale);
  }
  if (max_ofs % 10 == 0) {
    fmt::print("\n");
  } else {
    fmt::print("\n\n");
  }
}

// Debug print contiguous segment from index 0
void StatsUtil::printXSegment(float* x, int n_pts, float scale) {
  for (int i_ofs = 0; i_ofs < n_pts; i_ofs++) {
    if (i_ofs % 10 == 0) {
      fmt::print("\n{:6d}   ", i_ofs);
    }
    fmt::print("{:8.0f} ", x[i_ofs] * scale);
  }
  if (n_pts % 10 == 0) {
    fmt::print("\n");
  } else {
    fmt::print("\n\n");
  }
}

// Debug print segment with stride
void StatsUtil::printXSegmentStride(float* x, int n_pts, int stride, float scale) {
  for (int i_ofs = 0; i_ofs < n_pts; i_ofs++) {
    if (i_ofs % 10 == 0) {
      fmt::print("\n{:6d}   ", i_ofs * stride);
    }
    fmt::print("{:8.0f} ", x[i_ofs * stride] * scale);
  }
  if (n_pts % 10 == 0) {
    fmt::print("\n");
  } else {
    fmt::print("\n\n");
  }
}

// Debug print segment with frequency axis labels
void StatsUtil::printFXSegment(float* x, int n_pts, float scale, float f0, float df) {
  for (int i_ofs = 0; i_ofs < n_pts; i_ofs++) {
    if (i_ofs % 10 == 0) {
      fmt::print("\n{:6d} {:8.2f}  ", i_ofs, f0 + i_ofs * df);
    }
    fmt::print("{:8.0f} ", x[i_ofs] * scale);
  }
  if (n_pts % 10 == 0) {
    fmt::print("\n");
  } else {
    fmt::print("\n\n");
  }
}

// Debug print submatrix with per-row column drift shift
// A drifting tone should have a constant column in printout
void StatsUtil::printXSubmatrix(float* x, int n_row_x, int n_col_x, int start_row, int n_row,
                                int start_col, int n_col, float col_shift_per_row, float scale) {
  for (int i_row = start_row; i_row < start_row + n_row; i_row++) {
    int i_col_shift = static_cast<int>(round(i_row * col_shift_per_row));
    for (int i_col = 0; i_col < n_col; i_col++) {
      if (i_col % 10 == 0) {
        fmt::print("\n{:6d} {:6d}   ", i_row, start_col + i_col + i_col_shift);
      }
      fmt::print("{:8.0f} ", x[i_row * n_col_x + start_col + i_col + i_col_shift] * scale);
    }
    if (n_col > 10) {
      fmt::print("\n");
    }
  }
  fmt::print("\n\n");
}

// SubbandNormalizer: subband count and sigma-clipped stats

// Choose subband count from channel width (halve until min freq/subband met)
int SubbandNormalizer::chooseSubbandCount(int num_channels) {
  int n_subband = kNominalSubbands;
  int nf_subband = num_channels / n_subband;
  while ((nf_subband < kMinFreqPerSubband) || (n_subband == kMinSubbands)) {
    n_subband = std::max(kMinSubbands, n_subband / 2);
    nf_subband = num_channels / n_subband;
  }
  return n_subband;
}

// Per-subband mean/std, optionally sigma-clipped via subband_limit
void SubbandNormalizer::calcSubbandMeanStd(const float* spectrum, int num_channels, int n_subband,
                                           bool do_limit, float* subband_limit, float* work,
                                           float* subband_mean, float* subband_std) const {
  int nf_subband = num_channels / n_subband;

  for (int i_band = 0; i_band < n_subband; i_band++) {
    int i_ofs = i_band * nf_subband;
    if (do_limit) {
      float limit_value = subband_limit[i_band];
      for (int i = 0; i < nf_subband; i++) {
        work[i] = std::min(spectrum[i_ofs + i], limit_value);
      }
      StatsUtil::meanStdDev(work, nf_subband, &subband_mean[i_band], &subband_std[i_band]);
    } else {
      StatsUtil::meanStdDev(&spectrum[i_ofs], nf_subband, &subband_mean[i_band],
                            &subband_std[i_band]);
    }
  }
}

// Simplified three-pass sigma-clipped subband mean/std (limited at mean + sigma_clip_high_limit*std)
void SubbandNormalizer::multipassMeanStd(const float* spectrum, int num_channels, int n_subband,
                                         float sigma_clip_high_limit, float* work, float* subband_mean,
                                         float* subband_std, float* subband_limit) const {
  bool do_limit = false;
  calcSubbandMeanStd(spectrum, num_channels, n_subband, do_limit, subband_limit, work,
                     subband_mean, subband_std);

  for (int i_band = 0; i_band < n_subband; i_band++) {
    subband_limit[i_band] = subband_mean[i_band] + sigma_clip_high_limit * subband_std[i_band];
  }

  do_limit = true;
  calcSubbandMeanStd(spectrum, num_channels, n_subband, do_limit, subband_limit, work,
                     subband_mean, subband_std);

  for (int i_band = 0; i_band < n_subband; i_band++) {
    subband_limit[i_band] = subband_mean[i_band] + sigma_clip_high_limit * subband_std[i_band];
  }

  calcSubbandMeanStd(spectrum, num_channels, n_subband, do_limit, subband_limit, work,
                     subband_mean, subband_std);
}

SubbandNormalizer::SubbandNormalizer(int num_freq)
    : num_freq_(num_freq) {
  cudaMalloc(&gpu_subband_mean_, kNominalSubbands * sizeof(float));
  cudaMallocHost(&cpu_subband_mean_, kNominalSubbands * sizeof(float));
  checkCuda("subband_mean malloc");
  cudaMalloc(&gpu_subband_std_, kNominalSubbands * sizeof(float));
  cudaMallocHost(&cpu_subband_std_, kNominalSubbands * sizeof(float));
  checkCuda("subband_std malloc");
  cudaMalloc(&gpu_mu_std_work_, 3 * num_freq_ * sizeof(float));
  cudaMallocHost(&cpu_mu_std_work_, 3 * num_freq_ * sizeof(float));
  checkCuda("mu std work malloc");
}

SubbandNormalizer::~SubbandNormalizer() {
  cudaFree(gpu_subband_mean_);
  cudaFreeHost(cpu_subband_mean_);
  cudaFree(gpu_subband_std_);
  cudaFreeHost(cpu_subband_std_);
  cudaFree(gpu_mu_std_work_);
  cudaFreeHost(cpu_mu_std_work_);
}

// StampAnalyzer: per-column SK and line stats

// Per-column spectral kurtosis, SNR, and power stats on drift-shifted stamp 
// (spectrogram submatrix near detection)
void StampAnalyzer::computeSk(const float* stamp, int num_timesteps, int n_freq, float mu_noise,
                              float std_noise, int n_sti, int nbox, int start_row, int n_row,
                              int start_col, int n_col, float col_shift_per_row,
                              LineStats* lstats) {
  for (int i_col = 0; i_col < n_col; i_col++) {
    LineStats* lst = &lstats[i_col];
    lst->sk = 0.;
    lst->snr = 0.;
    lst->p_sum = 0.;
    lst->psq_sum = 0.;
    lst->p_max = -1e10;
    lst->p_min = 1e10;

    for (int i_row = start_row; i_row < start_row + n_row; i_row++) {
      int i_col_shift = static_cast<int>(round(i_row * col_shift_per_row));
      double temp = stamp[i_row * n_freq + start_col + i_col + i_col_shift];
      lst->p_sum += temp;
      lst->psq_sum += temp * temp;
      lst->p_max = std::max(temp, lst->p_max);
      lst->p_min = std::min(temp, lst->p_min);
    }

    double n_dof = 2. * n_sti * nbox;
    double p_sum_sq = lst->p_sum * lst->p_sum;
    lst->p_mean = lst->p_sum / n_row;
    lst->p_std = sqrt((lst->psq_sum - p_sum_sq / n_row) / (n_row - 1));
    lst->sk = (n_row * n_dof + 1.) / (n_row - 1) * (n_row * lst->psq_sum / p_sum_sq - 1.);
    lst->snr = (lst->p_mean - mu_noise) / std_noise * sqrt(nbox);
    lst->max_min_ratio = lst->p_max / lst->p_min;
  }
}

// StampAnalyzer: hit stamp debug dump

// Debug dump of hit stamp submatrix and LineStats row
void StampAnalyzer::printHitStampDebug(int coarse_channel, int hit_count, int candidate_freq,
                                       int drift_bins, int stamp_start_freq_idx, int hit_start_col,
                                       int hit_end_col, int hit_nbox, int hit_start_min,
                                       int hit_end_max, int stamp_width, int stamp_rows,
                                       float drift_bins_per_line, int n_stat_freqs,
                                       const LineStats* lstats) const {
  fmt::print("       stamp {} x {}: ifreq {} dbins {}, src start {} stamp {} - {}, Nbox {}, "
             "{} - {}\n\n",
             stamp_width, stamp_rows, candidate_freq, drift_bins, stamp_start_freq_idx,
             hit_start_col, hit_end_col, hit_nbox, hit_start_min, hit_end_max);
  int n_row = min(16, num_timesteps_);
  int n_col = 10;
  int start_col = hit_start_col - 2;
  fmt::print("Shifted stamp submatrix for coarse channel {}, hit {}, Nbox {}, bins/line "
             "{:.1f}, mid col {} (x10):\n",
             coarse_channel, hit_count, hit_nbox, drift_bins_per_line, hit_start_col);
  for (int i_row = 0; i_row < n_row; i_row++) {
    fmt::print("{:.0f} ", hit_start_col + drift_bins_per_line * i_row);
  }
  fmt::print("\n");
  StatsUtil::printXSubmatrix(cpu_stamp_, num_timesteps_, stamp_width, 0, n_row, start_col, n_col,
                             drift_bins_per_line, 10.f);
  fmt::print("SK =           ");
  for (int i = 0; i < n_stat_freqs; i++) {
    fmt::print("{:9.3f}", lstats[i].sk);
  }
  fmt::print("\n");
  fmt::print("SNR dB =       ");
  for (int i = 0; i < n_stat_freqs; i++) {
    fmt::print("{:9.1f}", 10.f * log10(lstats[i].snr));
  }
  fmt::print("\n");
  fmt::print("P_mean =       ");
  for (int i = 0; i < n_stat_freqs; i++) {
    fmt::print("{:9.2f}", lstats[i].p_mean);
  }
  fmt::print("\n");
  fmt::print("P_std =        ");
  for (int i = 0; i < n_stat_freqs; i++) {
    fmt::print("{:9.2f}", lstats[i].p_std);
  }
  fmt::print("\n");
  fmt::print("P_max =        ");
  for (int i = 0; i < n_stat_freqs; i++) {
    fmt::print("{:9.2f}", lstats[i].p_max);
  }
  fmt::print("\n");
  fmt::print("P_min =        ");
  for (int i = 0; i < n_stat_freqs; i++) {
    fmt::print("{:9.2f}", lstats[i].p_min);
  }
  fmt::print("\n");
  fmt::print("Max/Min =      ");
  for (int i = 0; i < n_stat_freqs; i++) {
    fmt::print("{:9.2f}", lstats[i].max_min_ratio);
  }
  fmt::print("\n\n");
}

StampAnalyzer::StampAnalyzer(int stamp_n_freq_max, int num_timesteps)
    : stamp_n_freq_max_(stamp_n_freq_max), num_timesteps_(num_timesteps) {
  cudaMalloc(&gpu_stamp_, stamp_n_freq_max_ * num_timesteps_ * sizeof(float));
  checkCuda("gpu_stamp_sg malloc");
  cudaMallocHost(&cpu_stamp_, stamp_n_freq_max_ * num_timesteps_ * sizeof(float));
  checkCuda("cpu_stamp_sg malloc");
}

StampAnalyzer::~StampAnalyzer() {
  cudaFree(gpu_stamp_);
  cudaFreeHost(cpu_stamp_);
}
