#pragma once

#include <string>
#include <vector>

#include "boxcar.h"
#include "dedoppler_hit.h"
#include "detection.h"
#include "filterbank_buffer.h"
#include "filterbank_metadata.h"
#include "hit_recorder.h"

using namespace std;

struct DedopplerConfig {
  static constexpr int kStampNFreqMax = 4096;
  static constexpr int kDcReplaceOfs = 15;
  static constexpr int kDcMeanPts = 40;
  static constexpr bool kDcReplaceEnable = true;

  BoxcarConfig boxcar;
};

class Dedopplerer {
 public:
  const int num_timesteps;
  const int num_channels;
  const double foff;
  const double tsamp;
  const bool has_dc_spike;
  bool print_hits;
  int debug = 0;

  Dedopplerer(int num_timesteps, int num_channels, double foff, double tsamp, bool has_dc_spike);
  ~Dedopplerer();

  void addIncoherentPower(const FilterbankBuffer& input, vector<DedopplerHit>& hits);

  void search(const FilterbankBuffer& input, const FilterbankMetadata& metadata, int beam,
              int coarse_channel, double max_drift, double min_drift, double snr_threshold,
              bool do_hit_screen, bool write_BB_hits_to_dat, vector<DedopplerHit>* output);

  size_t memoryUsage() const;

  Dedopplerer(const Dedopplerer&) = delete;
  Dedopplerer& operator=(Dedopplerer&) = delete;

 private:
  float* buffer1;
  float* buffer2;

  int rounded_num_timesteps;
  float* gpu_column_sums;
  float* cpu_column_sums;

  float* gpu_top_path_snrs;
  float* cpu_top_path_snrs;
  int* gpu_top_drift_blocks;
  int* cpu_top_drift_blocks;
  int* gpu_top_path_offsets;
  int* cpu_top_path_offsets;
  int* gpu_top_path_Nbox;
  int* cpu_top_path_Nbox;

  int drift_timesteps;
  double drift_rate_resolution;

  DedopplerConfig config_;
  BoxcarWorkspace boxcar_;
  SubbandNormalizer subband_;
  StampAnalyzer stamp_;
};
