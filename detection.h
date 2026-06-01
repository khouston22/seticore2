#pragma once

#include <vector>

#include "cuda_util.h"

using namespace std;

struct LineStats {
  double sk = 0.;
  double snr = 0.;
  double p_mean = 0.;
  double p_std = 0.;
  double p_sum = 0.;
  double psq_sum = 0.;
  double p_max = 0.;
  double p_min = 0.;
  double max_min_ratio = 0.;
};

class StatsUtil {
 public:
  static void meanStdDev(const float* x, int n, float* mean, float* std_dev);
  static void meanStdDev2(const float* x, int n, float* mean, float* std_dev);
  static float max(const float* x, int n);
  static float min(const float* x, int n);
  static void replaceDcSpike(float* x, int dc_ofs, int mean_pts);

  static void printXLr(float* x, int max_ofs, float scale);
  static void printXSegment(float* x, int n_pts, float scale);
  static void printXSegmentStride(float* x, int n_pts, int stride, float scale);
  static void printFXSegment(float* x, int n_pts, float scale, float f0, float df);
  static void printXSubmatrix(float* x, int n_row_x, int n_col_x, int start_row, int n_row,
                              int start_col, int n_col, float col_shift_per_row, float scale);
};

class SubbandNormalizer {
 public:
  static constexpr int kNominalSubbands = 128;
  static constexpr int kMinSubbands = 32;
  static constexpr int kMinFreqPerSubband = 4000;

  SubbandNormalizer(int num_freq);
  ~SubbandNormalizer();

  static int chooseSubbandCount(int num_channels);

  void calcSubbandMeanStd(const float* spectrum, int num_channels, int n_subband, bool do_limit,
                          float* subband_limit, float* work, float* subband_mean,
                          float* subband_std) const;
  void multipassMeanStd(const float* spectrum, int num_channels, int n_subband,
                        float sigma_clip_high_limit, float* work, float* subband_mean,
                        float* subband_std, float* subband_limit) const;

  void interpolateToFreqGpu(float* gpu_out, int n_freq, const float* gpu_subband, int n_subband);
  void equalizeSpectrogramRowGpu(float* d_sg_row, const float* gpu_mu, int n_freq);
  void computeSigmaScaleGpu(float* gpu_sigma_scale, const float* gpu_std, float nbox_gain,
                            int n_freq);

  float* gpuSubbandMean() { return gpu_subband_mean_; }
  float* gpuSubbandStd() { return gpu_subband_std_; }
  float* gpuMuStdWork() { return gpu_mu_std_work_; }
  float* cpuSubbandMean() { return cpu_subband_mean_; }
  float* cpuSubbandStd() { return cpu_subband_std_; }
  float* cpuMuStdWork() { return cpu_mu_std_work_; }

  SubbandNormalizer(const SubbandNormalizer&) = delete;
  SubbandNormalizer& operator=(const SubbandNormalizer&) = delete;

 private:
  int num_freq_;
  float* gpu_subband_mean_;
  float* gpu_subband_std_;
  float* gpu_mu_std_work_;
  float* cpu_subband_mean_;
  float* cpu_subband_std_;
  float* cpu_mu_std_work_;
};

class StampAnalyzer {
 public:
  StampAnalyzer(int stamp_n_freq_max, int num_timesteps);
  ~StampAnalyzer();

  static void computeSk(const float* stamp, int num_timesteps, int n_freq, float mu_noise,
                        float std_noise, int n_sti, int nbox, int start_row, int n_row,
                        int start_col, int n_col, float col_shift_per_row, LineStats* lstats);

  void printHitStampDebug(int coarse_channel, int hit_count, int candidate_freq, int drift_bins,
                          int stamp_start_column, int hit_start_mid, int hit_end_mid, int hit_nbox,
                          int hit_start_min, int hit_end_max, int stamp_width, int stamp_rows,
                          float drift_bins_per_line, int n_stat_freqs,
                          const LineStats* lstats) const;

  void extractStampGpu(float* dst, const float* src, int src_n_rows, int src_n_freq,
                       int start_freq, int n_freq_to_copy, int start_row, int n_rows_to_copy,
                       int nbox);

  float* gpuStamp() { return gpu_stamp_; }
  float* cpuStamp() { return cpu_stamp_; }
  int stampNFreqMax() const { return stamp_n_freq_max_; }

  StampAnalyzer(const StampAnalyzer&) = delete;
  StampAnalyzer& operator=(const StampAnalyzer&) = delete;

 private:
  int stamp_n_freq_max_;
  int num_timesteps_;
  float* gpu_stamp_;
  float* cpu_stamp_;
};
