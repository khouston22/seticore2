#include "filterbank_buffer.h"

#include <assert.h>
#include <fmt/core.h>
#include <iostream>

#include "cuda_util.h"
#include "util.h"

using namespace std;

// Creates a buffer that owns its own memory.
FilterbankBuffer::FilterbankBuffer(int num_timesteps, int num_channels)
  : num_timesteps(num_timesteps), num_channels(num_channels), managed(true),
    size(num_timesteps * num_channels),
    bytes(sizeof(float) * size) {
  cudaMallocManaged(&sg_data, bytes);
  cout << fmt::format("Filterbank managed data: {:.2f} MB\n",(long) bytes/1024./1024.);
  checkCudaMalloc("FilterbankBuffer", bytes);
  d_sg_data = sg_data;
}

// Creates a buffer that owns its own memory.
FilterbankBuffer::FilterbankBuffer(int num_timesteps, int num_channels, bool managed)
  : num_timesteps(num_timesteps), num_channels(num_channels), managed(managed),
    size(num_timesteps * num_channels),
    bytes(sizeof(float) * size) {
  if (managed) {
    cudaMallocManaged(&sg_data, bytes);
    cout << fmt::format("Filterbank managed2 data: {:.2f} MB\n",(long) bytes/1024./1024.);
    checkCudaMalloc("FilterbankBuffer managed", bytes);
    d_sg_data = sg_data;
  } else {
    cudaMallocHost(&sg_data, bytes);
    cout << fmt::format("Filterbank Host data: {:.2f} MB\n",(long) bytes/1024./1024.);
    checkCudaMalloc("FilterbankBuffer cudaMallocHost", bytes);
    cudaMalloc(&d_sg_data,bytes);
    checkCudaMalloc("cudaMalloc-d_sg_data",bytes);
    }
}

// Creates a buffer that is a view on memory owned by the caller.
FilterbankBuffer::FilterbankBuffer(int num_timesteps, int num_channels, float* sg_data)
  : num_timesteps(num_timesteps), num_channels(num_channels), managed(false),
    size(num_timesteps * num_channels),
    bytes(sizeof(float) * size), sg_data(sg_data) {
}

FilterbankBuffer::~FilterbankBuffer() {
  if (managed) {
    cudaFree(sg_data);
  } else {
    cudaFreeHost(sg_data);
    cudaFree(d_sg_data);
  }
}

// Set everything to zero
void FilterbankBuffer::zero() {
  memset(sg_data, 0, sizeof(float) * num_timesteps * num_channels);
  if (!managed) {
    // do explicit cpu to gpu copy for unmanaged sg buffers
    cudaMemcpy(d_sg_data,sg_data,bytes,cudaMemcpyHostToDevice);
    checkCuda("cudaMemcpy-d_sg");
  }
}

// Inefficient but useful for testing
// Note: for unmanaged buffer, sets host buffer only. Need to do cudaMemcpyHostToDevice afterward
void FilterbankBuffer::set(int time, int channel, float value) {
  assert(0 <= time && time < num_timesteps);
  assert(0 <= channel && channel < num_channels);
  int index = time * num_channels + channel;
  sg_data[index] = value;
}

// Note: for unmanaged buffer, gets host buffer value only. Need to do cudaMemcpyDeviceToHost beforehand
float FilterbankBuffer::get(int time, int channel) const {
  cudaDeviceSynchronize();
  checkCuda("FilterbankBuffer get");
  int index = time * num_channels + channel;
  return sg_data[index];
}

void FilterbankBuffer::assertEqual(const FilterbankBuffer& other, int drift_block) const {
  assert(num_timesteps == other.num_timesteps);
  assert(num_channels == other.num_channels);
  for (int drift = 0; drift < num_timesteps; ++drift) {
    for (int chan = 0; chan < num_channels; ++chan) {
      int last_chan = chan + (num_timesteps - 1) * drift_block + drift;
      if (last_chan < 0 || last_chan >= num_channels) {
        continue;
      }
      assertFloatEq(get(drift, chan), other.get(drift, chan),
                    fmt::format("data[{}][{}]", drift, chan));
    }
  }
}

// Make a filterbank buffer with a bit of deterministic noise so that
// normalization doesn't make everything infinite SNR.
FilterbankBuffer makeNoisyBuffer(int num_timesteps, int num_channels, bool managed) {
  FilterbankBuffer buffer(num_timesteps, num_channels, managed);
  buffer.zero();
  for (int chan = 0; chan < buffer.num_channels; ++chan) {
    buffer.set(0, chan, 0.1 * chan / buffer.num_channels);
  }
  if (!managed) {
    // do explicit cpu to gpu copy for unmanaged sg buffers
    cudaMemcpy(buffer.d_sg_data,buffer.sg_data,buffer.bytes,cudaMemcpyHostToDevice);
    checkCuda("cudaMemcpy-d_sg");
  }
  return buffer;
}

