#include <algorithm>
#include <assert.h>
#include <cuda.h>
#include <functional>
#include <iostream>
#include <math.h>
#include <numeric>
#include <vector>

#include "cuda_util.h"
#include "dedoppler.h"
#include "taylor.h"
#include "util.h"

#include "detection_fns.h"

/*
  Gather information about the top hits.

  The eventual goal is for every frequency freq, we want:

  top_path_sums[freq] to contain the largest path sum that starts at freq
  top_drift_blocks[freq] to contain the drift block of that path
  top_path_offsets[freq] to contain the path offset of that path

  path_sums[path_offset][freq] contains one path sum.
  (In row-major order.)
  So we are just taking the max along a column and carrying some
  metadata along as we find it. One thread per freq.

  The function ignores data corresponding to invalid paths. See
  comments in taylor.cu for details.
*/
__global__ void findTopPathSums(const float* path_sums, int num_timesteps, int num_freqs,
                                int drift_block, float* top_path_sums,
                                int* top_drift_blocks, int* top_path_offsets) {
  int freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (freq < 0 || freq >= num_freqs) {
    return;
  }

  for (int path_offset = 0; path_offset < num_timesteps; ++path_offset) {
    // Check if the last frequency in this path is out of bounds
    int last_freq = (num_timesteps - 1) * drift_block + path_offset + freq;
    if (last_freq < 0 || last_freq >= num_freqs) {
      // No more of these paths can be valid, either
      return;
    }

    float path_sum = path_sums[num_freqs * path_offset + freq];
    if (path_sum > top_path_sums[freq]) {
      top_path_sums[freq] = path_sum;
      top_drift_blocks[freq] = drift_block;
      top_path_offsets[freq] = path_offset;
    }
  }
}

/*
  Sum the columns of a two-dimensional array.
  input is a (num_timesteps x num_freqs) array, stored in row-major order.
  sums is an array of size num_freqs.
 */
__global__ void sumColumns(const float* input, float* sums, int num_timesteps, int num_freqs) {
  int freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (freq < 0 || freq >= num_freqs) {
    return;
  }
  sums[freq] = 0.0;
  for (int i = freq; i < num_timesteps * num_freqs; i += num_freqs) {
    sums[freq] += input[i];
  }
}


/*
  The Dedopplerer encapsulates the logic of dedoppler search. In particular it manages
  the needed GPU memory so that we can reuse the same memory allocation for different searches.
 */
Dedopplerer::Dedopplerer(int num_timesteps, int num_channels, double foff, double tsamp,
                         bool has_dc_spike)
    : num_timesteps(num_timesteps), num_channels(num_channels), foff(foff), tsamp(tsamp),
      has_dc_spike(has_dc_spike), print_hits(false) {
  assert(num_timesteps > 1);
  rounded_num_timesteps = roundUpToPowerOfTwo(num_timesteps);
  drift_timesteps = rounded_num_timesteps - 1;

  drift_rate_resolution = 1e6 * foff / (drift_timesteps * tsamp);
    
  // Allocate everything we need for GPU processing 
  cudaMalloc(&buffer1, num_channels * rounded_num_timesteps * sizeof(float));
  checkCuda("Dedopplerer buffer1 malloc");

  cudaMalloc(&buffer2, num_channels * rounded_num_timesteps * sizeof(float));
  checkCuda("Dedopplerer buffer2 malloc");
  
  cudaMalloc(&gpu_column_sums, num_channels * sizeof(float));
  cudaMallocHost(&cpu_column_sums, num_channels * sizeof(float));
  checkCuda("Dedopplerer column_sums malloc");
  
  cudaMalloc(&gpu_top_path_sums, num_channels * sizeof(float));
  cudaMallocHost(&cpu_top_path_sums, num_channels * sizeof(float));
  checkCuda("Dedopplerer top_path_sums malloc");
   
  cudaMalloc(&gpu_top_drift_blocks, num_channels * sizeof(int));
  cudaMallocHost(&cpu_top_drift_blocks, num_channels * sizeof(int));
  checkCuda("Dedopplerer top_drift_blocks malloc");
  
  cudaMalloc(&gpu_top_path_offsets, num_channels * sizeof(int));
  cudaMallocHost(&cpu_top_path_offsets, num_channels * sizeof(int));
  checkCuda("Dedopplerer top_path_offsets malloc");
}

Dedopplerer::~Dedopplerer() {
  cudaFree(buffer1);
  cudaFree(buffer2);
  cudaFree(gpu_column_sums);
  cudaFreeHost(cpu_column_sums);
  cudaFree(gpu_top_path_sums);
  cudaFreeHost(cpu_top_path_sums);
  cudaFree(gpu_top_drift_blocks);
  cudaFreeHost(cpu_top_drift_blocks);
  cudaFree(gpu_top_path_offsets);
  cudaFreeHost(cpu_top_path_offsets);
}

// This implementation is an ugly hack
size_t Dedopplerer::memoryUsage() const {
  return num_channels * rounded_num_timesteps * sizeof(float) * 2
    + num_channels * (2 * sizeof(float) + 2 * sizeof(int));
}

/*
  Takes a bunch of hits that we found for coherent beams, and adds information
  about their incoherent beam

  Input should be the incoherent sum.
  This function re-sorts hits by drift, so be aware that it will change order.
 */
void Dedopplerer::addIncoherentPower(const FilterbankBuffer& input,
                                     vector<DedopplerHit>& hits) {
  assert(input.num_timesteps == rounded_num_timesteps);
  assert(input.num_channels == num_channels);

  sort(hits.begin(), hits.end(), &driftStepsLessThan);
  
  int drift_shift = rounded_num_timesteps - 1;
  
  // The drift block we are currently analyzing
  int current_drift_block = INT_MIN;

  // A pointer for the currently-analyzed drift block
  const float* taylor_sums = nullptr;

  for (DedopplerHit& hit : hits) {
    // Figure out what drift block this hit belongs to
    int drift_block = (int) floor((float) hit.drift_steps / drift_shift);
    int path_offset = hit.drift_steps - drift_block * drift_shift;
    assert(0 <= path_offset && path_offset < drift_shift);

    // We should not go backwards
    assert(drift_block >= current_drift_block);

    if (drift_block > current_drift_block) {
      // We need to analyze a new drift block
      taylor_sums = optimizedTaylorTree(input.d_sg_data, buffer1, buffer2,
                                        rounded_num_timesteps, num_channels,
                                        drift_block);
      current_drift_block = drift_block;
    }

    long power_index = index2d(path_offset, hit.index, num_channels);
    assert(taylor_sums != nullptr);
    cudaMemcpy(&hit.incoherent_power, taylor_sums + power_index,
               sizeof(float), cudaMemcpyDeviceToHost);
  }
}

/*
  Runs dedoppler search on the input buffer.
  Output is appended to the output vector.
  
  All processing of the input buffer happens on the GPU, so it doesn't need to
  start off with host and device synchronized when search is called, it can still
  have GPU processing pending.
*/
void Dedopplerer::search(const FilterbankBuffer& input,
                         const FilterbankMetadata& metadata,
                         int beam, int coarse_channel,
                         double max_drift, double min_drift, double snr_threshold,
                         vector<DedopplerHit>* output) {
  assert(input.num_timesteps == rounded_num_timesteps);  // forces power of two
  assert(input.num_channels == num_channels);

  // Normalize the max drift in units of "horizontal steps per vertical step"
  double diagonal_drift_rate = drift_rate_resolution * drift_timesteps;
  double normalized_max_drift = max_drift / abs(diagonal_drift_rate);
  int min_drift_block = floor(-normalized_max_drift);
  int max_drift_block = floor(normalized_max_drift);

  int n_sti,n_lti,n_avg;
  float fs = metadata.foff*1e6; // FFT filter bank output sample rate prior to sti sum = bin bandwidth

  n_sti= MAX(1,abs(round(metadata.tsamp*fs)));
  n_lti = num_timesteps;
  n_avg = n_sti*n_lti;
  float xf = 1./n_avg/2.;

  int mid = num_channels / 2;

  printf("\ncoarse channel %d, FFT-size=%.0fK, n_sti=%d, n_lti=%d, n_avg=%d, Drift Blocks %d to %d\n",
          coarse_channel,num_channels/1024.,n_sti,n_lti,n_avg,min_drift_block,max_drift_block);

  long start_ms = timeInMS();
  long start_ms_all = timeInMS();
  
  // This will create one cuda thread per frequency bin
  int grid_size = (num_channels + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;

  if (!input.managed) {
    // do explicit cpu to gpu copy for unmanaged sg buffers
    cudaMemcpy(input.d_sg_data,input.sg_data,input.bytes,cudaMemcpyHostToDevice);
    checkCuda("cudaMemcpy-d_sg");
  }
 
  // Zero out the path sums in between each coarse channel because
  // we pick the top hits separately for each coarse channel
  cudaMemsetAsync(gpu_top_path_sums, 0, num_channels * sizeof(float));

  sumColumns<<<grid_size, CUDA_MAX_THREADS>>>(input.d_sg_data, gpu_column_sums,
                                              rounded_num_timesteps, num_channels);
  checkCuda("sumColumns");

  double t_sumcols_sec = (timeInMS() - start_ms)*.001;
  start_ms = timeInMS();

  // Do the Taylor tree algorithm for each drift block
  for (int drift_block = min_drift_block; drift_block <= max_drift_block; ++drift_block) {
    // Calculate Taylor sums
    const float* taylor_sums = optimizedTaylorTree(input.d_sg_data, buffer1, buffer2,
                                                   rounded_num_timesteps, num_channels,
                                                   drift_block);

    // Track the best sums
    findTopPathSums<<<grid_size, CUDA_MAX_THREADS>>>(taylor_sums, rounded_num_timesteps,
                                                     num_channels, drift_block,
                                                     gpu_top_path_sums,
                                                     gpu_top_drift_blocks,
                                                     gpu_top_path_offsets);
    checkCuda("findTopPathSums");
  }

  // Now that we have done all the GPU processing for one coarse
  // channel, we can copy the data back to host memory.
  // These copies are not async, so they will synchronize to the default stream.
  cudaMemcpy(cpu_column_sums, gpu_column_sums,
             num_channels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_top_path_sums, gpu_top_path_sums,
             num_channels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_top_drift_blocks, gpu_top_drift_blocks,
             num_channels * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_top_path_offsets, gpu_top_path_offsets,
             num_channels * sizeof(int), cudaMemcpyDeviceToHost);
  checkCuda("dedoppler d->h memcpy");
  
  double t_DD_sec = (timeInMS() - start_ms)*.001;
  
  /*
  ** Run special test averaging increasing durations, verify non-coh gain
  */

  #if 0
    ncoh_avg_test(input.sg_data, num_channels, num_timesteps, n_sti, 1);
    ncoh_avg_test(input.sg_data, num_channels, num_timesteps, n_sti, 8);
    ncoh_avg_test(input.sg_data, num_channels, num_timesteps, n_sti, 32);
    ncoh_avg_test(input.sg_data, num_channels, num_timesteps, n_sti, 128);
  #endif

  /*
  ** Find detections
  */

  start_ms = timeInMS();

  // normalize by number of averages
  for (int freq=0; freq<num_channels; freq++){
    cpu_column_sums[freq] *= xf;
    cpu_top_path_sums[freq] *= xf;
  }

  #if 0
    printf("Column sums: DC vicinity:");
    print_x_lr(&cpu_column_sums[mid],100,1.0);
  #endif
 
  #if DC_REPLACE_ENABLE
    DC_replace(&cpu_column_sums[mid], DC_REPLACE_OFS, DC_MEAN_PTS);
  #endif

  #if 0
    printf("Column sums: DC vicinity after replacement:");
    print_x_lr(&cpu_column_sums[mid],100,1.0);
  #endif

  int n_subband = N_SUBBAND;
  int Nf_subband = num_channels/n_subband;
  if (Nf_subband<NF_SUBBAND_MIN) {
    n_subband = MAX(1,num_channels/NF_SUBBAND_MIN);
    Nf_subband = num_channels/n_subband;
  }
  float subband_mean[N_SUBBAND_MAX];
  float subband_std[N_SUBBAND_MAX];
  float subband_limit[N_SUBBAND_MAX];
  float subband_det_threshold[N_SUBBAND_MAX];
  float subband_m_std_ratio[N_SUBBAND_MAX];
  float *work;
  work = (float *) malloc(Nf_subband*sizeof(float));
  printf("\nFFT-size=%.0fK, n_subband=%d, Nf_subband=%d => %.0f Hz/subband:\n",num_channels/1024.,n_subband,
        Nf_subband,Nf_subband*fs);

  if (n_subband==1){
    calc_mean_std_dev(cpu_column_sums,num_channels,&subband_mean[0],&subband_std[0]);
  } else {
    float shear_constant = 2.3;
    multipass_subband_mean_std(cpu_column_sums,num_channels,n_subband,shear_constant,
                work,subband_mean,subband_std,subband_limit);
  }
  
  for (int i_band=0; i_band<n_subband; i_band++){
    subband_m_std_ratio[i_band] = subband_mean[i_band]/subband_std[i_band];
    subband_det_threshold[i_band] = subband_mean[i_band] + snr_threshold*subband_std[i_band];
    if (i_band==0) {
        printf("subband=%d mean=%.0f std=%.0f mean/std=%.2f vs %.2f, snr_threshold=%.2f, det_thr=%.0f\n",
        i_band,subband_mean[i_band],subband_std[i_band],subband_m_std_ratio[i_band],sqrt(2*n_avg),
        snr_threshold,subband_det_threshold[i_band]);
    }
  }
  // examine mean/std ratio mean and std over all subbands
  // should have low variation (std should be small)
  float m_std_ratio_mean,m_std_ratio_std;
  if (n_subband>1) { 
    calc_mean_std_dev(subband_m_std_ratio,n_subband,&m_std_ratio_mean,&m_std_ratio_std);
    printf("n_subband=%d m_std_ratio_mean=%.2f vs %.2f m_std_ratio_std=%.2f\n\n",
          n_subband,m_std_ratio_mean,sqrt(2*n_avg),m_std_ratio_std);
  }
  
  float m,std_dev;
  calc_mean_std_dev(cpu_column_sums, num_channels, &m, &std_dev);

  double t_stats_sec = (timeInMS() - start_ms)*.001;
  start_ms = timeInMS();
    
  // We consider two hits to be duplicates if the distance in their
  // frequency indexes is less than window_size. We only want to
  // output the largest representative of any set of duplicates.
  // window_size is chosen just large enough so that a single bright
  // pixel cannot cause multiple hits.
  // First we break up the data into a set of nonoverlapping
  // windows. Any candidate hit must be the largest within this
  // window.
  int window_size = 2 * ceil(normalized_max_drift * drift_timesteps);

  if (coarse_channel==0) {
    printf("foff=%f MHz t_samp=%f sec, n_sti=%d, n_lti=%d, n_avg=%d, n_fft=%d\n",
            metadata.foff*1e6,metadata.tsamp,n_sti,n_lti,n_avg,num_channels);
    printf("drift_rate_resolution=%.3f drift_timesteps=%d diagonal_drift_rate=%.3f\n",
            drift_rate_resolution,drift_timesteps,diagonal_drift_rate);
    printf("max_drift=%.2f normalized_max_drift=%.2f drift_timesteps=%d window_size=%d=>%.0f Hz\n",
            max_drift,normalized_max_drift,drift_timesteps,window_size,window_size*fs);
    printf("Overall Coarse Channel mean=%6.0f std_dev=%6.0f mean/std=%6.3f vs %6.3f\n\n",
            m,std_dev,m/std_dev,sqrt(2*n_avg));
  }

  for (int i = 0; i * window_size < num_channels; ++i) {
    int candidate_freq = -1;

    int i_band = MIN(n_subband-1,((i+0.5) * window_size)/Nf_subband);
    float path_sum_threshold = subband_det_threshold[i_band];
    float local_mean = subband_mean[i_band];
    std_dev = subband_std[i_band];
  
    float candidate_path_sum = path_sum_threshold;

    for (int j = 0; j < window_size; ++j) {
      int freq = i * window_size + j;
      if (freq >= num_channels) {
        break;
      }
      if (cpu_top_path_sums[freq] > candidate_path_sum) {
        // This is the new best candidate of the window
        candidate_freq = freq;
        candidate_path_sum = cpu_top_path_sums[freq];
      }
    }
    if (candidate_freq < 0) {
      continue;
    }

    // Check every frequency closer than window_size if we have a candidate
    int window_end = min(num_channels, candidate_freq + window_size);
    bool found_larger_path_sum = false;
    for (int freq = max(0, candidate_freq - window_size + 1); freq < window_end; ++freq) {
      if (cpu_top_path_sums[freq] > candidate_path_sum) {
        found_larger_path_sum = true;
        break;
      }
    }
    if (!found_larger_path_sum) {
      // The candidate frequency is the best within its window
      int drift_bins = cpu_top_drift_blocks[candidate_freq] * drift_timesteps +
        cpu_top_path_offsets[candidate_freq];
      double drift_rate = drift_bins * drift_rate_resolution;
      float snr = (candidate_path_sum - local_mean) / std_dev;

      // Hack: to look at snr components for detections using test files, just set them to snr
      // float snr = candidate_path_sum/1e3;
      // float snr = local_mean/10;
      // float snr = std_dev/10;

      if ((abs(drift_rate) >= min_drift) && (abs(drift_rate)) <= max_drift) {
        DedopplerHit hit(metadata, candidate_freq, drift_bins, drift_rate,
                         snr, beam, coarse_channel, num_timesteps, candidate_path_sum);
        if (print_hits) {
          cout << "hit: " << hit.toString() << endl;
        }
        output->push_back(hit);
      }
    }
  }

  free(work);
  
  double t_log_hits_sec = (timeInMS() - start_ms)*.001;
  double t_search_sec = (timeInMS() - start_ms_all)*.001;

  printf("Elapsed times: coarse chnl %d, UM %d, fft %d, sti %d, lti %d\n",
              coarse_channel,(int)input.managed,num_channels,n_sti,n_lti);
  printf("Sum Columns:     %.3f sec\n",t_sumcols_sec);
  printf("Taylor GPU:      %.3f sec\n",t_DD_sec);
  printf("Stats:           %.3f sec\n",t_stats_sec);
  printf("Log Hits:        %.3f sec\n",t_log_hits_sec);
  printf("DeDoppler total: %.3f sec\n",t_search_sec);

}
