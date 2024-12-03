#pragma once

#include <cstddef>

#define MANAGED_INPUT 0

using namespace std;

/*
  The FilterbankBuffer stores the contents of a filterbank file in unified memory.
  Just one beam. This can be a single coarse channel, or the entire file.
 */
class FilterbankBuffer {
 public:
  const int num_timesteps;
  const int num_channels;

  // Whether the buffer owns its own memory
  const bool managed;

  const int size;
  const size_t bytes;
  
  /*
    Row-major indexed by:
      sg_data[time][freq]
   */
  // host and device spectrogram data (managed) or host pinned sg data (unmanaged)
  float* sg_data; 
  // spectrogram data in gpu device (managed or unmanaged) 
  float* d_sg_data;  

  // Create a managed buffer
  FilterbankBuffer(int num_timesteps, int num_channels);
  // Create a managed buffer or host-only buffer
  FilterbankBuffer(int num_timesteps, int num_channels, bool managed);

  // Create an unmanaged buffer, essentially a view on a pre-existing buffer
  FilterbankBuffer(int num_timesteps, int num_channels, float* sg_data);
  
  ~FilterbankBuffer();

  // Only implicit moving, no implicit copying
  FilterbankBuffer(const FilterbankBuffer&) = delete;
  FilterbankBuffer& operator=(FilterbankBuffer&) = delete;
  FilterbankBuffer(FilterbankBuffer&& other) = default;
  
  void zero();
  void set(int time, int channel, float value);
  float get(int time, int channel) const;

  // Assert two filterbanks are equal over the indexes that are valid for
  // this drift block.
  void assertEqual(const FilterbankBuffer& other, int drift_block) const;
};

// Fill a buffer with meaningless data for testing
FilterbankBuffer makeNoisyBuffer(int num_timesteps, int num_channels, bool managed);

