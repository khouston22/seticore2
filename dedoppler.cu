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

#define LOG2_MAX_NBOX_P2 (6)  // determines maximum memory reqts for boxcar averaging of DD sums
#include "boxcar_sum.h"

#include "detection_fns.h"

/*
  Gather information about the top hits.

  The eventual goal is for every frequency freq, we want:

  top_path_snrs[freq] to contain the largest path snr that starts at freq
  top_drift_blocks[freq] to contain the drift block of that path
  top_path_offsets[freq] to contain the path offset of that path
  top_path_Nbox[freq] to contain the value of Nbox that path

  path_sums[path_offset][freq] contains one path sum.
  (In row-major order.)
  So we are just taking the max along a column and carrying some
  metadata along as we find it. One thread per freq.

  The function ignores data corresponding to invalid paths. See
  comments in taylor.cu for details.
*/

__global__ void findTopPathSNRs(const float* path_sums, int num_timesteps, int num_freqs,
                                int drift_block, float mu, float* sigma_scale, int Nbox,
                                float* top_path_snrs, int* top_drift_blocks, int* top_path_offsets,
                                int* top_path_Nbox) {

  int freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (freq < 0 || freq >= num_freqs) {
    return;
  }
  // mu is nominal mean after equalization across band 
  // sigma_scale[freq] = sqrt(Nbox)/sigma[freq] typically
  float path_scale = 1./num_timesteps;  // assumes DD algorithm does not normalize by #lines summed

  for (int path_offset = 0; path_offset < num_timesteps; ++path_offset) {
  
    // Check if the first or last frequency in this path is out of bounds
    if (drift_block>=0) {
      int last_freq = num_freqs - 1 - ((num_timesteps - 1) * drift_block + path_offset) - Nbox;
      if (freq > last_freq) {
        return;
      }
    } else {
      int first_freq = -((num_timesteps - 1) * drift_block + path_offset) + Nbox;
      if (freq < first_freq) {
        return;
      }
    }

    // // Incorrect approach
    // // Check if the last frequency in this path is out of bounds
    // int last_freq = (num_timesteps - 1) * drift_block + path_offset + freq;
    // if (last_freq < 0 || last_freq >= num_freqs) {
    //   // No more of these paths can be valid, either
    //   return;
    // }

    float path_snr = (path_sums[num_freqs * path_offset + freq]*path_scale - mu)*sigma_scale[freq];
    if (path_snr > top_path_snrs[freq]) {
      top_path_snrs[freq] = path_snr;
      top_drift_blocks[freq] = drift_block;
      top_path_offsets[freq] = path_offset;
      top_path_Nbox[freq] = Nbox;
    }
  }
}

__global__ void findTopPathSNRs_1step(const float* path_sums_line, int num_timesteps, int num_freqs,
                                int path_offset, int drift_block, float mu, float* sigma_scale, int Nbox,
                                float* top_path_snrs, int* top_drift_blocks, int* top_path_offsets,
                                int* top_path_Nbox) {

  int freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (freq < 0 || freq >= num_freqs) {
    return;
  }
  // examines single time step (single line) of DD output after boxcar filtering
  // and updates peak snr for each frequency
  // mu is nominal mean after equalization across band 
  // sigma_scale[freq] = sqrt(Nbox)/sigma[freq] typically
  float path_scale = 1./num_timesteps;  // assumes DD algorithm does not normalize by #lines summed

  // Check if the first or last frequency in this path is out of bounds
  if (drift_block>=0) {
    int last_freq = num_freqs - 1 - ((num_timesteps - 1) * drift_block + path_offset) - Nbox;
    if (freq > last_freq) {
      return;
    }
  } else {
    int first_freq = -((num_timesteps - 1) * drift_block + path_offset) + Nbox;
    if (freq < first_freq) {
      return;
    }
  }

  float path_snr = (path_sums_line[freq]*path_scale - mu)*sigma_scale[freq];
  if (path_snr > top_path_snrs[freq]) {
    top_path_snrs[freq] = path_snr;
    top_drift_blocks[freq] = drift_block;
    top_path_offsets[freq] = path_offset;
    top_path_Nbox[freq] = Nbox;
  }
}


/*
  Sum the columns of a two-dimensional array.
  input is a (num_timesteps x num_freqs) array, stored in row-major order.
  sums is an array of size num_freqs.
 */
__global__ void sumColumns(const float* input, float* sums, int num_timesteps, int num_freqs,
                            float scale) {
  int freq = blockIdx.x * blockDim.x + threadIdx.x;
  if (freq < 0 || freq >= num_freqs) {
    return;
  }
  sums[freq] = 0.0;
  for (int i = freq; i < num_timesteps * num_freqs; i += num_freqs) {
    sums[freq] += input[i];
  }
  sums[freq] *= scale;
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
  
  cudaMalloc(&gpu_top_path_snrs, num_channels * sizeof(float));
  cudaMallocHost(&cpu_top_path_snrs, num_channels * sizeof(float));
  checkCuda("Dedopplerer top_path_snrs malloc");
   
  cudaMalloc(&gpu_top_drift_blocks, num_channels * sizeof(int));
  cudaMallocHost(&cpu_top_drift_blocks, num_channels * sizeof(int));
  checkCuda("Dedopplerer top_drift_blocks malloc");
  
  cudaMalloc(&gpu_top_path_offsets, num_channels * sizeof(int));
  cudaMallocHost(&cpu_top_path_offsets, num_channels * sizeof(int));
  checkCuda("Dedopplerer top_path_offsets malloc");

  cudaMalloc(&gpu_top_path_Nbox, num_channels * sizeof(int));
  cudaMallocHost(&cpu_top_path_Nbox, num_channels * sizeof(int));
  checkCuda("Dedopplerer top_path_Nbox malloc");

  int num_channels_ext = num_channels + 2*N_ZP;
  cudaMalloc(&gpu_p2_path_sums, num_channels_ext*N_P2*sizeof(float));
  checkCuda("p2_path_sums malloc");
  cudaMalloc(&gpu_boxcar_work, 2*num_channels_ext*sizeof(float));
  // cudaMallocHost(&cpu_boxcar_work, 2*num_channels_ext*sizeof(float));
  checkCuda("boxcar work malloc");

  cudaMalloc(&gpu_Nbox_path_sum, num_channels*sizeof(float));
  checkCuda("gpu_Nbox_path_sum malloc");

  cudaMalloc(&gpu_mu_std_work, 3*num_channels*sizeof(float));
  cudaMallocHost(&cpu_mu_std_work, 3*num_channels*sizeof(float));
  checkCuda("mu std work malloc");

  cudaMalloc(&gpu_subband_mean, N_SUBBAND_MAX*sizeof(float));
  cudaMallocHost(&cpu_subband_mean, N_SUBBAND_MAX*sizeof(float));
  checkCuda("subband_mean malloc");

  cudaMalloc(&gpu_subband_std, N_SUBBAND_MAX*sizeof(float));
  cudaMallocHost(&cpu_subband_std, N_SUBBAND_MAX*sizeof(float));
  checkCuda("subband_std malloc");
}

Dedopplerer::~Dedopplerer() {
  cudaFree(buffer1);
  cudaFree(buffer2);
  cudaFree(gpu_column_sums);
  cudaFreeHost(cpu_column_sums);
  cudaFree(gpu_top_path_snrs);
  cudaFreeHost(cpu_top_path_snrs);
  cudaFree(gpu_top_drift_blocks);
  cudaFreeHost(cpu_top_drift_blocks);
  cudaFree(gpu_top_path_offsets);
  cudaFreeHost(cpu_top_path_offsets);
  cudaFree(gpu_top_path_Nbox);
  cudaFreeHost(cpu_top_path_Nbox);

  cudaFree(gpu_p2_path_sums);
  cudaFree(gpu_boxcar_work);
  // cudaFreeHost(cpu_boxcar_work);

  cudaFree(gpu_Nbox_path_sum);
  
  cudaFree(gpu_mu_std_work);
  cudaFreeHost(cpu_mu_std_work);

  cudaFree(gpu_subband_mean);
  cudaFreeHost(cpu_subband_mean);
  cudaFree(gpu_subband_std);
  cudaFreeHost(cpu_subband_std);
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
  float f_ctr_MHz = metadata.fch1 + (coarse_channel+0.5)*metadata.foff*num_channels;
  
  n_sti= MAX(1,abs(round(metadata.tsamp*fs)));
  n_lti = num_timesteps;
  n_avg = n_sti*n_lti;
  
  int mid = num_channels / 2;

  printf("\ncoarse channel %d: %.3f MHz, FFT-size=%.0fK, n_sti=%d, n_lti=%d, n_avg=%d, Drift Blocks %d to %d\n",
          coarse_channel,f_ctr_MHz,num_channels/1024.,n_sti,n_lti,n_avg,min_drift_block,max_drift_block);

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
  cudaMemsetAsync(gpu_top_path_snrs, 0, num_channels * sizeof(float));

  // cudaDeviceSynchronize();
  double t_input_copy_sec = (timeInMS() - start_ms)*.001;
  start_ms = timeInMS();

  float scale = 1./num_timesteps;
  sumColumns<<<grid_size, CUDA_MAX_THREADS>>>(input.d_sg_data, gpu_column_sums,
                                              num_timesteps, num_channels, scale);
  checkCuda("sumColumns");
  cudaMemcpy(cpu_column_sums, gpu_column_sums,
            num_channels * sizeof(float), cudaMemcpyDeviceToHost);
  checkCuda("sumColumns d->h memcpy");
  // cudaDeviceSynchronize();
  
  double t_sumcols_sec = (timeInMS() - start_ms)*.001;
  start_ms = timeInMS();

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

  // Copy DC points from CPU to GPU
  cudaMemcpy(&gpu_column_sums[mid-DC_REPLACE_OFS], &cpu_column_sums[mid-DC_REPLACE_OFS],
            (2*DC_REPLACE_OFS+1) * sizeof(float), cudaMemcpyHostToDevice);
  checkCuda("sumColumns DC h->d memcpy");
  // Copy DC points from CPU to GPU Spectrograms d_sg_data
  for (int i_time=0; i_time<num_timesteps; i_time++) {
    cudaMemcpy(&input.d_sg_data[i_time*num_channels+mid-DC_REPLACE_OFS], &cpu_column_sums[mid-DC_REPLACE_OFS],
            (2*DC_REPLACE_OFS+1) * sizeof(float), cudaMemcpyHostToDevice);
  }
  checkCuda("d_sg_data DC h->d memcpy");

  /*
  ** Compute mean & std for subbands, first pass
  */
  
  int n_subband = N_SUBBAND;
  int Nf_subband = num_channels/n_subband;
  if (Nf_subband<NF_SUBBAND_MIN) {
    n_subband = MAX(1,num_channels/NF_SUBBAND_MIN);
    Nf_subband = num_channels/n_subband;
  }
  float subband_limit[N_SUBBAND_MAX];
  float shear_constant = 2.3;
  float *subband_work;
  subband_work = (float *) malloc(num_channels*sizeof(float));  // allow for max size for one subband

  printf("\nFFT-size=%.0fK, n_subband=%d, Nf_subband=%d => %.0f Hz/subband:\n",num_channels/1024.,n_subband,
        Nf_subband,Nf_subband*fs);

  multipass_subband_mean_std(cpu_column_sums,num_channels,n_subband,shear_constant,
                subband_work,cpu_subband_mean,cpu_subband_std,subband_limit);
 
  // Check overall mean & std with just one subband (entire coarse channel)
  float mu,std_dev;
  multipass_subband_mean_std(cpu_column_sums,num_channels,1,shear_constant,
                subband_work,&mu,&std_dev,subband_limit);
  printf("Coarse Channel %d Single Subband mean=%6.3f std_dev=%6.3f mean/std=%6.3f vs %6.3f\n\n",
            coarse_channel,mu,std_dev,mu/std_dev,sqrt(2*n_avg));

  /*
  ** Run special test averaging increasing durations, verify non-coh gain
  */

  #if 0
    ncoh_avg_test(input.sg_data, num_channels, num_timesteps, n_sti, 1);
    ncoh_avg_test(input.sg_data, num_channels, num_timesteps, n_sti, 8);
    ncoh_avg_test(input.sg_data, num_channels, num_timesteps, n_sti, 32);
    ncoh_avg_test(input.sg_data, num_channels, num_timesteps, n_sti, 128);
  #endif

  #if 0
    // if (coarse_channel==0) {
      printf("n_subband=%d mean values:\n",n_subband);
      print_x_segment(cpu_subband_mean, n_subband, 1.0);
      printf("n_subband=%d std  values:\n",n_subband);
      print_x_segment(cpu_subband_std , n_subband, 1.0);
    // }
  #endif

  double t_stats_sec = (timeInMS() - start_ms)*.001;
  start_ms = timeInMS();
 
  /*
  ** Scale input data in GPU to unit mean by interpolation, recompute subband stats, scale again
  */

  cudaMemcpy(gpu_subband_mean,cpu_subband_mean,n_subband*sizeof(float),cudaMemcpyHostToDevice);
  checkCuda("cudaMemcpy-gpu_subband_mean");
  float* gpu_mu_vector = &gpu_mu_std_work[0];
  gpu_subband_interpolate<<<grid_size, CUDA_MAX_THREADS>>>(gpu_mu_vector, 
                                  num_channels, gpu_subband_mean, n_subband);
  
  #if 0
    if (coarse_channel==0) {
      float* cpu_mu_vector = &cpu_mu_std_work[0];
      cudaMemcpy(cpu_mu_vector,gpu_mu_vector,num_channels*sizeof(float),cudaMemcpyDeviceToHost);
      checkCuda("cudaMemcpy-mu_vector");
      printf("\nn_subband=%d interpolated mean values:\n",n_subband);
      print_x_segment_stride(cpu_mu_vector, n_subband, Nf_subband, 1.0);
    }
  #endif


  for (int i_time=0; i_time < num_timesteps; i_time++) {
    gpu_local_mean_scale<<<grid_size, CUDA_MAX_THREADS>>>(&input.d_sg_data[i_time*num_channels], 
                                        gpu_mu_vector, num_channels);
  }

  float scale2 = 1./num_timesteps;
  sumColumns<<<grid_size, CUDA_MAX_THREADS>>>(input.d_sg_data, gpu_column_sums,
                                              num_timesteps, num_channels, scale2);
  checkCuda("sumColumns");
  cudaMemcpy(cpu_column_sums, gpu_column_sums,
            num_channels * sizeof(float), cudaMemcpyDeviceToHost);
  checkCuda("sumColumns d->h memcpy");
  
  multipass_subband_mean_std(cpu_column_sums,num_channels,n_subband,shear_constant,
              subband_work,cpu_subband_mean,cpu_subband_std,subband_limit);

  #if 1
    // if (coarse_channel==0) {
      printf("n_subband=%d mean values after scale (x1000):\n",n_subband);
      print_x_segment(cpu_subband_mean, n_subband, 1000.0);
      printf("n_subband=%d std  values after scale (x1000):\n",n_subband);
      print_x_segment(cpu_subband_std , n_subband, 1000.0);
    // }
  #endif
  

  // overall mean & std dev once normalized

  multipass_subband_mean_std(cpu_column_sums,num_channels,1,shear_constant,
                subband_work,&mu,&std_dev,subband_limit);
  printf("Coarse Channel %d Multi-subband mean=%6.3f std_dev=%6.3f mean/std=%6.3f vs %6.3f\n\n",
            coarse_channel,mu,std_dev,mu/std_dev,sqrt(2*n_avg));


  // generate revised interpolated mu, std, sigma_scale vectors

  cudaMemcpy(gpu_subband_mean,cpu_subband_mean,n_subband*sizeof(float),cudaMemcpyHostToDevice);
  checkCuda("cudaMemcpy-gpu_subband_mean");
  cudaMemcpy(gpu_subband_std ,cpu_subband_std ,n_subband*sizeof(float),cudaMemcpyHostToDevice);
  checkCuda("cudaMemcpy-gpu_subband_mean");

  gpu_mu_vector = &gpu_mu_std_work[0];
  gpu_subband_interpolate<<<grid_size, CUDA_MAX_THREADS>>>(gpu_mu_vector, 
                                  num_channels, gpu_subband_mean, n_subband);
  float* gpu_std_vector = &gpu_mu_std_work[num_channels];
  gpu_subband_interpolate<<<grid_size, CUDA_MAX_THREADS>>>(gpu_std_vector, 
                                  num_channels, gpu_subband_std, n_subband);
  float* gpu_sigma_scale_vector= &gpu_mu_std_work[2*num_channels];
  float sqrtNbox = 1.;
  gpu_compute_sigma_scale<<<grid_size, CUDA_MAX_THREADS>>>(gpu_sigma_scale_vector, 
                                gpu_std_vector, sqrtNbox, num_channels);

  #if 0
    if (coarse_channel==0) {
      float* cpu_mu_vector = &cpu_mu_std_work[0];
      cudaMemcpy(cpu_mu_vector,gpu_mu_vector,num_channels*sizeof(float),cudaMemcpyDeviceToHost);
      checkCuda("cudaMemcpy-mu_vector");
      printf("\nn_subband=%d interpolated mean values (x1000):\n",n_subband);
      print_x_segment_stride(cpu_mu_vector, n_subband, Nf_subband, 1000.0);
      float* cpu_std_vector = &cpu_mu_std_work[num_channels];
      cudaMemcpy(cpu_std_vector,gpu_std_vector,num_channels*sizeof(float),cudaMemcpyDeviceToHost);
      checkCuda("cudaMemcpy-std_vector");
      printf("\nn_subband=%d interpolated std values (x1000):\n",n_subband);
      print_x_segment_stride(cpu_std_vector, n_subband, Nf_subband, 1000.0);
      float* cpu_sigma_scale_vector = &cpu_mu_std_work[2*num_channels];
      cudaMemcpy(cpu_sigma_scale_vector,gpu_sigma_scale_vector,num_channels*sizeof(float),cudaMemcpyDeviceToHost);
      checkCuda("cudaMemcpy-sigma_scale_vector");
      printf("\nn_subband=%d interpolated sigma_scale values:\n",n_subband);
      print_x_segment_stride(cpu_sigma_scale_vector, n_subband, Nf_subband, 1.0);
    }
  #endif

  double t_scale_sec = (timeInMS() - start_ms)*.001;
  start_ms = timeInMS();

  /*
  ** De-Doppler
  */

  // set up boxcar average parameters

  int log2_max_p2;
  int n_zp = N_ZP;

  int n_Nbox,Nbox_max;

  int max_Nbox_bw = 1;    // only one Nbox value according to drift block
  // int max_Nbox_bw = 2; // one Nbox value, except Nbox = 1 and 2 for blocks -1 and 0
  // int max_Nbox_bw = 32; // many Nbox values, min ~abs(drift_block), up to 32
  
  int* Nbox_list = (int *) malloc((LOG2_MAX_NBOX_P2+1)*sizeof(int));
  max_Nbox_bw = MIN(max_Nbox_bw,NBOX_P2_MAX);


  // Do the Taylor tree algorithm for each drift block

  for (int drift_block = min_drift_block; drift_block <= max_drift_block; ++drift_block) {

    // Calculate Taylor sums
    const float* taylor_sums = optimizedTaylorTree(input.d_sg_data, buffer1, buffer2,
                                                   rounded_num_timesteps, num_channels,
                                                   drift_block);

    // Do boxcar filtering for each line of taylor sums and update SNR values

    #define DO_BOXCAR 1
    #if DO_BOXCAR
      n_Nbox = gen_Nbox_list1(Nbox_list, drift_block, max_Nbox_bw);
    #else
      Nbox_list[0] = 1;
      n_Nbox = 1;
    #endif

    Nbox_max = Nbox_list[n_Nbox-1];
    log2_max_p2 = (int) floor(log2(Nbox_max));
    
    if ((coarse_channel==0) && (drift_block==0)) {
      print_Nbox_list(Nbox_list,n_Nbox,drift_block);
    }
    
    for (int i_Nbox=0; i_Nbox<n_Nbox; i_Nbox++) {

      int Nbox = Nbox_list[i_Nbox];
      
      float sqrtNbox = sqrt(Nbox);
      gpu_compute_sigma_scale<<<grid_size, CUDA_MAX_THREADS>>>(gpu_sigma_scale_vector, 
                                gpu_std_vector, sqrtNbox, num_channels);

      for (int path_offset = 0; path_offset < rounded_num_timesteps; ++path_offset) {

        float *gpu_Nbox_path_sum_line;
        bool init_boxcar = false;

        if (Nbox == 1) {
          gpu_Nbox_path_sum_line = (float *) &taylor_sums[path_offset*num_channels];
        } else {
          // Do the boxcar filtering 

          // Generate power of 2 sum vectors
          if (!init_boxcar) {
            gpu_DD_sums_line = (float *) &taylor_sums[path_offset*num_channels];
            gen_boxcar_p2_sums_gpu(gpu_p2_path_sums,gpu_DD_sums_line,num_channels,log2_max_p2,n_zp);
            checkCuda("cudaMemcpy-gen_boxcar_p2_sums_gpu");
            init_boxcar = true;
          }
          // Generate boxcar sums for arbitrary Nbox values
          
          gen_boxcar_sum_gpu(gpu_Nbox_path_sum,gpu_p2_path_sums,gpu_boxcar_work,Nbox,num_channels,log2_max_p2,n_zp);
          gpu_Nbox_path_sum_line = gpu_Nbox_path_sum;
          
        }
        // Update the best SNRs over frequency - one line at a time
        findTopPathSNRs_1step<<<grid_size, CUDA_MAX_THREADS>>>(gpu_Nbox_path_sum_line, rounded_num_timesteps,
                                num_channels, path_offset, drift_block, mu, gpu_sigma_scale_vector, Nbox,
                                gpu_top_path_snrs, gpu_top_drift_blocks, gpu_top_path_offsets,
                                gpu_top_path_Nbox);
        checkCuda("findTopPathSNRs_1step");
      }
    }

    // Update the best SNRs over frequency
    
    #if 0
      // uses local estimate of spectrum std dev to estimate SNR
      findTopPathSNRs<<<grid_size, CUDA_MAX_THREADS>>>(taylor_sums, rounded_num_timesteps,
                                num_channels, drift_block, mu, gpu_sigma_scale_vector, Nbox,
                                gpu_top_path_snrs, gpu_top_drift_blocks, gpu_top_path_offsets,
                                gpu_top_path_Nbox);
      checkCuda("findTopPathSNRs");
    #endif
  }

  // Now that we have done all the GPU processing for one coarse
  // channel, we can copy the data back to host memory.
  // These copies are not async, so they will synchronize to the default stream.
  // cudaMemcpy(cpu_column_sums, gpu_column_sums,
  //            num_channels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_top_path_snrs, gpu_top_path_snrs,
             num_channels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_top_drift_blocks, gpu_top_drift_blocks,
             num_channels * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_top_path_offsets, gpu_top_path_offsets,
             num_channels * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_top_path_Nbox, gpu_top_path_Nbox,
             num_channels * sizeof(int), cudaMemcpyDeviceToHost);
  checkCuda("dedoppler d->h memcpy");
  

  double t_DD_sec = (timeInMS() - start_ms)*.001;
  
  /*
  ** Find detections
  */

  start_ms = timeInMS();

     
  // We consider two hits to be duplicates if the distance in their
  // frequency indexes is less than window_size. We only want to
  // output the largest representative of any set of duplicates.
  // window_size is chosen just large enough so that a single bright
  // pixel cannot cause multiple hits.
  // First we break up the data into a set of nonoverlapping
  // windows. Any candidate hit must be the largest within this
  // window.
  // Original seticore 1.0.6:
  int window_size = 2 * ceil(normalized_max_drift * drift_timesteps);
  // Minimum window size to avoid extra spurious detections on single drifting tone:
  // int window_size = 1 * ceil(normalized_max_drift * drift_timesteps);
  // Will be proportional to max drift rate and total averaging time
  // Will also determine allowable spacing between adjacent hits
  // May want to set window size in Hz at the command line depending on RFI environment
  // to avoid multiple hits on same RFI signal

  if (coarse_channel==0) {
    printf("foff=%f MHz t_samp=%f sec, n_sti=%d, n_lti=%d, n_avg=%d, n_fft=%d\n",
            metadata.foff*1e6,metadata.tsamp,n_sti,n_lti,n_avg,num_channels);
    printf("drift_rate_resolution=%.3f drift_timesteps=%d diagonal_drift_rate=%.3f\n",
            drift_rate_resolution,drift_timesteps,diagonal_drift_rate);
    printf("max_drift=%.2f normalized_max_drift=%.2f drift_timesteps=%d window_size=%d=>%.0f Hz\n\n",
            max_drift,normalized_max_drift,drift_timesteps,window_size,window_size*fs);
  }

  for (int i = 0; i * window_size < num_channels; ++i) {
    int candidate_freq = -1;

    int i_band = MIN(n_subband-1,((i+0.5) * window_size)/Nf_subband);
    std_dev = cpu_subband_std[i_band];
  
    float candidate_path_snr = snr_threshold;

    for (int j = 0; j < window_size; ++j) {
      int freq = i * window_size + j;
      if (freq >= num_channels) {
        break;
      }
      if (cpu_top_path_snrs[freq] > candidate_path_snr) {
        // This is the new best candidate of the window
        candidate_freq = freq;
        candidate_path_snr = cpu_top_path_snrs[freq];
      }
    }
    if (candidate_freq < 0) {
      continue;
    }

    // Check every frequency closer than window_size if we have a candidate
    int window_end = min(num_channels, candidate_freq + window_size);
    bool found_larger_path_snr = false;
    for (int freq = max(0, candidate_freq - window_size + 1); freq < window_end; ++freq) {
      if (cpu_top_path_snrs[freq] > candidate_path_snr) {
        found_larger_path_snr = true;
        break;
      }
    }
    if (!found_larger_path_snr) {
      // The candidate frequency is the best within its window
      int drift_bins = cpu_top_drift_blocks[candidate_freq] * drift_timesteps +
        cpu_top_path_offsets[candidate_freq];
      double drift_rate = drift_bins * drift_rate_resolution;
      float snr = candidate_path_snr;
      float drift_tol = .05;

      if ((abs(drift_rate) >= min_drift) && (abs(drift_rate)) <= max_drift+drift_tol) {
        DedopplerHit hit(metadata, candidate_freq, drift_bins, drift_rate,
                         snr, beam, coarse_channel, num_timesteps, candidate_path_snr);
        float f_hit_MHz = metadata.fch1 + (coarse_channel*num_channels+candidate_freq)*metadata.foff;
        float snr_db = 10*log10(snr);
        if (print_hits) {
          printf("hit: chnl %d, %8d %10.3f MHz %6.2f Hz/sec, %5.2f dB SNR\n",
                  coarse_channel,candidate_freq-num_channels/2,f_hit_MHz, drift_rate, snr_db);
          // cout << "hit: " << hit.toString() << endl;
        }
        output->push_back(hit);
      }
    }
  }

  free(subband_work);
  free(Nbox_list);
  
  
  double t_log_hits_sec = (timeInMS() - start_ms)*.001;
  double t_search_sec = (timeInMS() - start_ms_all)*.001;

  printf("\nElapsed times: coarse chnl %d, UM %d, fft %d, sti %d, lti %d\n",
              coarse_channel,(int)input.managed,num_channels,n_sti,n_lti);
  printf("Input copy:      %.3f sec\n",t_input_copy_sec);
  printf("Sum Columns:     %.3f sec\n",t_sumcols_sec);
  printf("Stats:           %.3f sec\n",t_stats_sec);
  printf("Scale input:     %.3f sec\n",t_scale_sec);
  printf("Taylor GPU:      %.3f sec\n",t_DD_sec);
  printf("Log Hits:        %.3f sec\n",t_log_hits_sec);
  printf("DeDoppler total: %.3f sec\n",t_search_sec);

}
