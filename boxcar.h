#pragma once

#include <vector>

#include "cuda_util.h"

using namespace std;

struct BoxcarConfig {
  static constexpr int kDefaultLog2MaxP2 = 6;

  int log2_max_p2 = kDefaultLog2MaxP2;

  int nbox_p2_max() const { return 1 << log2_max_p2; }
  int nbox_max() const { return 2 * nbox_p2_max() - 1; }
  int n_zp() const { return 2 * nbox_p2_max(); }
  int n_p2() const { return log2_max_p2 + 1; }
};

class BoxcarWorkspace {
 public:
  BoxcarWorkspace(int num_freq, const BoxcarConfig& config);
  ~BoxcarWorkspace();

  int numFreq() const { return num_freq_; }
  const BoxcarConfig& config() const { return config_; }

  float* gpuNboxPathSum() { return gpu_nbox_path_sum_; }
  float* gpuP2PathSums() { return gpu_p2_path_sums_; }

  const float* filterLineGpu(const float* dd_sums_line, int nbox, int log2_max_p2_runtime);

  void computeP2SumsCpu(const float* dd_sums_line, int log2_max_p2_runtime,
                        float* p2_path_sums, int n_zp) const;
  void computeSumCpu(const float* p2_path_sums, float* work, int nbox,
                     int log2_max_p2_runtime, int n_zp, float* nbox_path_sum) const;

  void computeP2SumsGpu(const float* dd_sums_line, int log2_max_p2_runtime, int n_zp);
  void computeSumGpu(int nbox, int log2_max_p2_runtime, int n_zp);

  static vector<int> buildNboxList(int drift_block, int max_nbox_bw, int nbox_p2_max);
  static void printNboxList(const vector<int>& nbox_list, int drift_block);
  static void printNboxSegment(const float* x, int n_pts, int start_offset, float scale);

  BoxcarWorkspace(const BoxcarWorkspace&) = delete;
  BoxcarWorkspace& operator=(const BoxcarWorkspace&) = delete;

 private:
  int num_freq_;
  BoxcarConfig config_;
  float* gpu_p2_path_sums_;
  float* gpu_boxcar_work_;
  float* gpu_nbox_path_sum_;
};
