#pragma once

#include <cuda.h>
#include <string>
#include <vector>

#include "dedoppler_hit.h"
#include "filterbank_buffer.h"
#include "filterbank_metadata.h"
#include "hit_recorder.h"

using namespace std;


class Dedopplerer {
public:
  // How many timesteps of data are processed by the engine.
  // They may be stored in a larger array.
  const int num_timesteps;

  // How many frequency channels of data are processed by the engine.
  const int num_channels;

  // Frequency difference between adjacent bins, in MHz
  const double foff;

  // Time difference between adjacent bins, in seconds
  const double tsamp;

  // Whether the data we receive has a DC spike
  const bool has_dc_spike;

  // // Whether to apply nominal screening against hits likely to be RFI
  // bool do_hit_screen;

  // // Whether to write broadband detections to dat file
  // bool write_BB_hits_to_dat;

  bool print_hits;
  
  // Do not round num_timesteps before creating the Dedopplerer
  Dedopplerer(int num_timesteps, int num_channels, double foff, double tsamp,
              bool has_dc_spike);
  ~Dedopplerer();

  void addIncoherentPower(const FilterbankBuffer& input, vector<DedopplerHit>& hits);
  
  void search(const FilterbankBuffer& input, const FilterbankMetadata& metadata,
              int beam, int coarse_channel,
              double max_drift, double min_drift, double snr_threshold,
              bool do_hit_screen, bool write_BB_hits_to_dat,
              vector<DedopplerHit>* output);

  size_t memoryUsage() const;

  // No copying
  Dedopplerer(const Dedopplerer&) = delete;
  Dedopplerer& operator=(Dedopplerer&) = delete;
  
private:
  // For computing Taylor sums, we use three on-gpu arrays,
  // each the size of one coarse channel. The input array is to read the source
  // data, and these two buffers are to use for Taylor tree calculations.
  float *buffer1, *buffer2;

  // The size of the larger array that contains the number of timesteps,
  // typically rounded up to the nearest power of two.
  int rounded_num_timesteps;
  
  // For normalization, we sum each column.
  float *gpu_column_sums, *cpu_column_sums;

  // For aggregating top hits, we use three arrays, each the size of
  // one row of the coarse channel. One array to store the largest
  // path sum, one to store which drift block it came from, and one to
  // store its path offset.
  float *gpu_top_path_snrs, *cpu_top_path_snrs;
  int *gpu_top_drift_blocks, *cpu_top_drift_blocks;
  int *gpu_top_path_offsets, *cpu_top_path_offsets;
  int *gpu_top_path_Nbox, *cpu_top_path_Nbox;

  // How many timesteps the signal drifts in our data
  int drift_timesteps;

  // The difference in adjacent drift rates that we look for, in Hz/s
  double drift_rate_resolution;  

  // subband mean and std values
  float *gpu_subband_mean, *cpu_subband_mean;
  float *gpu_subband_std, *cpu_subband_std;
  
  // interpolated mean and std values work
  float *gpu_mu_std_work, *cpu_mu_std_work;
  
  // boxcar averaging pointers to gpu memory blocks
  float *gpu_DD_sums_line, *gpu_p2_path_sums, *gpu_boxcar_work, *gpu_Nbox_path_sum;
  float *cpu_boxcar_work;

  // stamp submatrix centered on hit
  float *gpu_stamp_sg,*cpu_stamp_sg;
  

};

