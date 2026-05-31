#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#include "boxcar.h"
#include "broadband_detector.h"
#include "dedoppler.h"
#include "dedoppler_kernels.h"
#include "taylor.h"
#include "util.h"

using namespace std;

Dedopplerer::Dedopplerer(int num_timesteps, int num_channels, double foff, double tsamp,
                         bool has_dc_spike)
    : num_timesteps(num_timesteps),
      num_channels(num_channels),
      foff(foff),
      tsamp(tsamp),
      has_dc_spike(has_dc_spike),
      print_hits(false),
      debug(0),
      rounded_num_timesteps(roundUpToPowerOfTwo(num_timesteps)),
      drift_timesteps(roundUpToPowerOfTwo(num_timesteps) - 1),
      config_(),
      boxcar_(num_channels, config_.boxcar),
      subband_(num_channels),
      stamp_(DedopplerConfig::kStampNFreqMax, roundUpToPowerOfTwo(num_timesteps)) {
  assert(num_timesteps > 1);

  drift_rate_resolution = 1e6 * foff / (drift_timesteps * tsamp);

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
}

size_t Dedopplerer::memoryUsage() const {
  return num_channels * rounded_num_timesteps * sizeof(float) * 2
         + num_channels * (2 * sizeof(float) + 2 * sizeof(int));
}

void Dedopplerer::addIncoherentPower(const FilterbankBuffer& input, vector<DedopplerHit>& hits) {
  assert(input.num_timesteps == rounded_num_timesteps);
  assert(input.num_channels == num_channels);

  sort(hits.begin(), hits.end(), &driftStepsLessThan);

  int drift_shift = rounded_num_timesteps - 1;
  int current_drift_block = INT_MIN;
  const float* taylor_sums = nullptr;

  for (DedopplerHit& hit : hits) {
    int drift_block = static_cast<int>(floor(static_cast<float>(hit.drift_steps) / drift_shift));
    int path_offset = hit.drift_steps - drift_block * drift_shift;
    assert(0 <= path_offset && path_offset < drift_shift);
    assert(drift_block >= current_drift_block);

    if (drift_block > current_drift_block) {
      taylor_sums = optimizedTaylorTree(input.d_sg_data, buffer1, buffer2, rounded_num_timesteps,
                                        num_channels, drift_block);
      current_drift_block = drift_block;
    }

    long power_index = index2d(path_offset, hit.index, num_channels);
    assert(taylor_sums != nullptr);
    cudaMemcpy(&hit.incoherent_power, taylor_sums + power_index, sizeof(float),
               cudaMemcpyDeviceToHost);
  }
}

void Dedopplerer::search(const FilterbankBuffer& input, const FilterbankMetadata& metadata,
                         int beam, int coarse_channel, double max_drift, double min_drift,
                         double snr_threshold, bool do_hit_screen, bool write_BB_hits_to_dat,
                         vector<DedopplerHit>* output) {
  assert(input.num_timesteps == rounded_num_timesteps);
  assert(input.num_channels == num_channels);

  double diagonal_drift_rate = drift_rate_resolution * drift_timesteps;
  double normalized_max_drift = max_drift / abs(diagonal_drift_rate);
  int min_drift_block = floor(-normalized_max_drift);
  int max_drift_block = floor(normalized_max_drift);

  int n_sti, n_lti, n_avg;
  float fs = metadata.foff * 1e6;
  float f1_MHz = metadata.fch1 + (coarse_channel * num_channels) * metadata.foff;
  float f2_MHz = metadata.fch1 + (coarse_channel * num_channels + num_channels - 1) * metadata.foff;

  n_sti = max(1, abs(static_cast<int>(round(metadata.tsamp * fs))));
  n_lti = num_timesteps;
  n_avg = n_sti * n_lti;

  int mid = num_channels / 2;
  int hit_count = 0;
  int stamp_print_count = 0;

  printf("\ncoarse channel %d: %.3f-%.3f MHz, FFT-size=%.0fK, n_sti=%d, n_lti=%d, n_avg=%d, "
         "Drift Blocks %d to %d\n",
         coarse_channel, f1_MHz, f2_MHz, num_channels / 1024., n_sti, n_lti, n_avg,
         min_drift_block, max_drift_block);

  long start_ms = timeInMS();
  long start_ms_all = timeInMS();

  if (!input.managed) {
    cudaMemcpy(input.d_sg_data, input.sg_data, input.bytes, cudaMemcpyHostToDevice);
    checkCuda("cudaMemcpy-d_sg");
  }

  cudaMemsetAsync(gpu_top_path_snrs, 0, num_channels * sizeof(float));
  double t_input_copy_sec = (timeInMS() - start_ms) * .001;
  start_ms = timeInMS();

  float scale = 1.f / num_timesteps;
  launchSumColumns(input.d_sg_data, gpu_column_sums, num_timesteps, num_channels, scale);
  cudaMemcpy(cpu_column_sums, gpu_column_sums, num_channels * sizeof(float),
             cudaMemcpyDeviceToHost);
  checkCuda("sumColumns d->h memcpy");

  double t_sumcols_sec = (timeInMS() - start_ms) * .001;
  start_ms = timeInMS();

  if (debug >= 3) {
    printf("Column sums: DC vicinity:");
    StatsUtil::printXLr(&cpu_column_sums[mid], 100, 1.0);
  }

  if (DedopplerConfig::kDcReplaceEnable) {
    StatsUtil::replaceDcSpike(&cpu_column_sums[mid], DedopplerConfig::kDcReplaceOfs,
                              DedopplerConfig::kDcMeanPts);
  }

  if (debug >= 3) {
    printf("Column sums: DC vicinity after replacement:");
    StatsUtil::printXLr(&cpu_column_sums[mid], 100, 1.0);
  }

  cudaMemcpy(&gpu_column_sums[mid - DedopplerConfig::kDcReplaceOfs],
             &cpu_column_sums[mid - DedopplerConfig::kDcReplaceOfs],
             (2 * DedopplerConfig::kDcReplaceOfs + 1) * sizeof(float), cudaMemcpyHostToDevice);
  checkCuda("sumColumns DC h->d memcpy");

  for (int i_time = 0; i_time < num_timesteps; i_time++) {
    cudaMemcpy(&input.d_sg_data[i_time * num_channels + mid - DedopplerConfig::kDcReplaceOfs],
               &cpu_column_sums[mid - DedopplerConfig::kDcReplaceOfs],
               (2 * DedopplerConfig::kDcReplaceOfs + 1) * sizeof(float), cudaMemcpyHostToDevice);
  }
  checkCuda("d_sg_data DC h->d memcpy");

  int n_subband = SubbandNormalizer::chooseSubbandCount(num_channels);
  int nf_subband = num_channels / n_subband;
  if (nf_subband < SubbandNormalizer::kMinFreqPerSubband) {
    printf("Warning: #subbands=%d, freq bins per subband=%d vs. %d desired\n", n_subband,
           nf_subband, SubbandNormalizer::kMinFreqPerSubband);
  }

  float subband_limit[SubbandNormalizer::kNominalSubbands];
  float shear_constant = 3.0f;
  float* subband_work = static_cast<float*>(malloc(num_channels * sizeof(float)));
  float f0_sb_MHz = metadata.fch1 + (coarse_channel * num_channels + nf_subband / 2) * metadata.foff;
  float df_sb_MHz = nf_subband * metadata.foff;

  float* cpu_subband_mean = subband_.cpuSubbandMean();
  float* cpu_subband_std = subband_.cpuSubbandStd();

  printf("\nFFT-size=%.0fK, n_subband=%d, Nf_subband=%d => %.0f Hz/subband:\n",
         num_channels / 1024., n_subband, nf_subband, nf_subband * fs);

  subband_.multipassMeanStd(cpu_column_sums, num_channels, n_subband, shear_constant, subband_work,
                            cpu_subband_mean, cpu_subband_std, subband_limit);

  float subband_mean0[SubbandNormalizer::kNominalSubbands];
  memcpy(subband_mean0, cpu_subband_mean, n_subband * sizeof(float));

  float mu, std_dev;
  subband_.multipassMeanStd(cpu_column_sums, num_channels, 1, shear_constant, subband_work, &mu,
                            &std_dev, subband_limit);
  printf("Coarse Channel %d Single Subband mean=%6.3f std_dev=%6.3f mean/std=%6.3f vs %6.3f\n\n",
         coarse_channel, mu, std_dev, mu / std_dev, sqrt(2 * n_avg));

  double t_stats_sec = (timeInMS() - start_ms) * .001;
  start_ms = timeInMS();

  cudaMemcpy(subband_.gpuSubbandMean(), cpu_subband_mean, n_subband * sizeof(float),
             cudaMemcpyHostToDevice);
  checkCuda("cudaMemcpy-gpu_subband_mean");
  float* gpu_mu_vector = subband_.gpuMuStdWork();
  subband_.interpolateToFreqGpu(gpu_mu_vector, num_channels, subband_.gpuSubbandMean(), n_subband);

  if (debug >= 4 && coarse_channel == 0) {
    float* cpu_mu_vector = subband_.cpuMuStdWork();
    cudaMemcpy(cpu_mu_vector, gpu_mu_vector, num_channels * sizeof(float),
               cudaMemcpyDeviceToHost);
    checkCuda("cudaMemcpy-mu_vector");
    printf("\nn_subband=%d interpolated mean values:\n", n_subband);
    StatsUtil::printXSegmentStride(cpu_mu_vector, n_subband, nf_subband, 1.0);
  }

  for (int i_time = 0; i_time < num_timesteps; i_time++) {
    subband_.equalizeSpectrogramRowGpu(&input.d_sg_data[i_time * num_channels], gpu_mu_vector,
                                       num_channels);
  }

  float scale2 = 1.f / num_timesteps;
  launchSumColumns(input.d_sg_data, gpu_column_sums, num_timesteps, num_channels, scale2);
  cudaMemcpy(cpu_column_sums, gpu_column_sums, num_channels * sizeof(float),
             cudaMemcpyDeviceToHost);
  checkCuda("sumColumns d->h memcpy");

  subband_.multipassMeanStd(cpu_column_sums, num_channels, n_subband, shear_constant, subband_work,
                            cpu_subband_mean, cpu_subband_std, subband_limit);

  float subband_mean_no_clip[SubbandNormalizer::kNominalSubbands];
  float subband_std_no_clip[SubbandNormalizer::kNominalSubbands];
  subband_.calcSubbandMeanStd(cpu_column_sums, num_channels, n_subband, false, subband_limit,
                              subband_work, subband_mean_no_clip, subband_std_no_clip);

  float blk_sk_clip[SubbandNormalizer::kNominalSubbands];
  float blk_sk_no_clip[SubbandNormalizer::kNominalSubbands];
  float blk_sk_clip_mean, blk_sk_clip_std;
  float blk_sk_no_clip_mean, blk_sk_no_clip_std;

  for (int i_subband = 0; i_subband < n_subband; i_subband++) {
    blk_sk_clip[i_subband] =
        (2 * nf_subband * n_avg + 1.) / nf_subband *
        pow(cpu_subband_std[i_subband] / cpu_subband_mean[i_subband], 2.0);
    blk_sk_no_clip[i_subband] =
        (2 * nf_subband * n_avg + 1.) / nf_subband *
        pow(subband_std_no_clip[i_subband] / subband_mean_no_clip[i_subband], 2.0);
  }
  StatsUtil::meanStdDev(blk_sk_clip, n_subband, &blk_sk_clip_mean, &blk_sk_clip_std);
  StatsUtil::meanStdDev(blk_sk_no_clip, n_subband, &blk_sk_no_clip_mean, &blk_sk_no_clip_std);

  if (debug >= 1 && coarse_channel == 0) {
    printf("chnl %d n_subband=%d sigma clipped mean values after scale (x1000):\n", coarse_channel,
          n_subband);
    StatsUtil::printXSegment(cpu_subband_mean, n_subband, 1000.0);
    printf("chnl %d n_subband=%d sigma clipped std  values after scale (x1000):\n", coarse_channel,
          n_subband);
    StatsUtil::printFXSegment(cpu_subband_std, n_subband, 1000.0, f0_sb_MHz, df_sb_MHz);
  }
  
  if (debug >= 2 && coarse_channel == 0) {
    printf("chnl %d n_subband=%d no clip std  values after scale (x1000):\n", coarse_channel,
           n_subband);
    StatsUtil::printFXSegment(subband_std_no_clip, n_subband, 1000.0, f0_sb_MHz, df_sb_MHz);
  }

  float subband_std_mean_nominal = 1.0f / sqrt(2 * n_avg);
  float subband_std_mean_norm[SubbandNormalizer::kNominalSubbands];
  for (int i_subband = 0; i_subband < n_subband; i_subband++) {
    subband_std_mean_norm[i_subband] =
        cpu_subband_std[i_subband] / cpu_subband_mean[i_subband] / subband_std_mean_nominal;
  }

  if (debug >= 1) {
    printf("chnl %d n_subband=%d sigma clipped std/mean values over expected after scale (x100):\n",
          coarse_channel, n_subband);
    StatsUtil::printFXSegment(subband_std_mean_norm, n_subband, 100.0, f0_sb_MHz, df_sb_MHz);

    printf("chnl %d n_subband=%d clipped SK  values after scale (x100), mean=%.3f, std=%.3f:\n",
          coarse_channel, n_subband, blk_sk_clip_mean, blk_sk_clip_std);
    StatsUtil::printFXSegment(blk_sk_clip, n_subband, 100.0, f0_sb_MHz, df_sb_MHz);
    printf("chnl %d n_subband=%d no clip SK  values after scale (x100), mean=%.3f, std=%.3f:\n",
          coarse_channel, n_subband, blk_sk_no_clip_mean, blk_sk_no_clip_std);
    StatsUtil::printFXSegment(blk_sk_no_clip, n_subband, 100.0, f0_sb_MHz, df_sb_MHz);
  }

  subband_.multipassMeanStd(cpu_column_sums, num_channels, 1, shear_constant, subband_work, &mu,
                            &std_dev, subband_limit);
  printf("Coarse Channel %d Multi-subband mean=%6.3f std_dev=%6.3f mean/std=%6.3f vs %6.3f\n\n",
        coarse_channel, mu, std_dev, mu / std_dev, sqrt(2 * n_avg));
  

  float bb_z_det = 5.f;
  float bb_det_threshold = 1.02f / sqrt(2 * n_avg) * (1.f + bb_z_det / sqrt(nf_subband));
  float bb_det_threshold_sk = pow(bb_det_threshold, 2.0) * 2 * n_avg;
  float* subband_std_bb_det = subband_std_no_clip;
  float* blk_sk = blk_sk_no_clip;

  int n_subband_dilation = 3;
  BroadbandDetector bb_detector;
  bb_detector.BroadbandDetect(n_subband, nf_subband, n_subband_dilation, debug, bb_det_threshold, bb_det_threshold_sk,
                              f0_sb_MHz, df_sb_MHz, subband_std_bb_det, blk_sk, cpu_column_sums,
                              cpu_subband_mean, cpu_subband_std);

  if (write_BB_hits_to_dat) {
    for (int i_bb_det = 0; i_bb_det < bb_detector.nDetections(); i_bb_det++) {
      const BBdet& det = bb_detector.detections()[i_bb_det];
      int freq_idx = det.sb1 * nf_subband;
      DedopplerHit hit(metadata, freq_idx, det.fctr_MHz, det.f1_MHz, det.f2_MHz, 0, 0., det.snr, 0,
                       coarse_channel, num_timesteps, 0., det.peak_blk_sk, -1., -1.);
      output->push_back(hit);
    }
  }

  for (int i_subband = 0; i_subband < n_subband; i_subband++) {
    if (bb_detector.subbandDetected()[i_subband] > 0.f) {
      cpu_subband_std[i_subband] =
          subband_std_no_clip[i_subband] / subband_mean_no_clip[i_subband];
    }
  }

  cudaMemcpy(subband_.gpuSubbandMean(), cpu_subband_mean, n_subband * sizeof(float),
             cudaMemcpyHostToDevice);
  checkCuda("cudaMemcpy-gpu_subband_mean");
  cudaMemcpy(subband_.gpuSubbandStd(), cpu_subband_std, n_subband * sizeof(float),
             cudaMemcpyHostToDevice);
  checkCuda("cudaMemcpy-gpu_subband_std");

  gpu_mu_vector = subband_.gpuMuStdWork();
  subband_.interpolateToFreqGpu(gpu_mu_vector, num_channels, subband_.gpuSubbandMean(), n_subband);
  float* gpu_std_vector = &subband_.gpuMuStdWork()[num_channels];
  subband_.interpolateToFreqGpu(gpu_std_vector, num_channels, subband_.gpuSubbandStd(), n_subband);
  float* gpu_sigma_scale_vector = &subband_.gpuMuStdWork()[2 * num_channels];
  subband_.computeSigmaScaleGpu(gpu_sigma_scale_vector, gpu_std_vector, 1.f, num_channels);

  if (debug >= 2 && coarse_channel == 0) {
    float* cpu_mu_vector = subband_.cpuMuStdWork();
    cudaMemcpy(cpu_mu_vector, gpu_mu_vector, num_channels * sizeof(float),
               cudaMemcpyDeviceToHost);
    checkCuda("cudaMemcpy-mu_vector");
    printf("\nn_subband=%d interpolated mean values (x1000):\n", n_subband);
    StatsUtil::printXSegmentStride(cpu_mu_vector, n_subband, nf_subband, 1000.0);
    float* cpu_std_vector = &subband_.cpuMuStdWork()[num_channels];
    cudaMemcpy(cpu_std_vector, gpu_std_vector, num_channels * sizeof(float),
               cudaMemcpyDeviceToHost);
    checkCuda("cudaMemcpy-std_vector");
    printf("\nn_subband=%d interpolated std values (x1000):\n", n_subband);
    StatsUtil::printXSegmentStride(cpu_std_vector, n_subband, nf_subband, 1000.0);
    float* cpu_sigma_scale_vector = &subband_.cpuMuStdWork()[2 * num_channels];
    cudaMemcpy(cpu_sigma_scale_vector, gpu_sigma_scale_vector, num_channels * sizeof(float),
               cudaMemcpyDeviceToHost);
    checkCuda("cudaMemcpy-sigma_scale_vector");
    printf("\nn_subband=%d interpolated sigma_scale values:\n", n_subband);
    StatsUtil::printXSegmentStride(cpu_sigma_scale_vector, n_subband, nf_subband, 1.0);
  }

  double t_scale_sec = (timeInMS() - start_ms) * .001;
  start_ms = timeInMS();

  int n_zp = config_.boxcar.n_zp();
  int max_nbox_bw = 1;
  max_nbox_bw = min(max_nbox_bw, config_.boxcar.nbox_p2_max());

  for (int drift_block = min_drift_block; drift_block <= max_drift_block; ++drift_block) {
    const float* taylor_sums = optimizedTaylorTree(input.d_sg_data, buffer1, buffer2,
                                                   rounded_num_timesteps, num_channels, drift_block);

    vector<int> nbox_list =
        BoxcarWorkspace::buildNboxList(drift_block, max_nbox_bw, config_.boxcar.nbox_p2_max());
    int nbox_max = nbox_list.back();
    int log2_max_p2 = static_cast<int>(floor(log2(nbox_max)));

    if ((coarse_channel == 0) && (drift_block == 0)) {
      BoxcarWorkspace::printNboxList(nbox_list, drift_block);
    }

    for (int nbox : nbox_list) {
      float nbox_gain = pow(nbox, .40);
      subband_.computeSigmaScaleGpu(gpu_sigma_scale_vector, gpu_std_vector, nbox_gain, num_channels);

      for (int path_offset = 0; path_offset < rounded_num_timesteps; ++path_offset) {
        const float* gpu_nbox_path_sum_line;
        if (nbox == 1) {
          gpu_nbox_path_sum_line = &taylor_sums[path_offset * num_channels];
        } else {
          const float* gpu_dd_sums_line = &taylor_sums[path_offset * num_channels];
          boxcar_.computeP2SumsGpu(gpu_dd_sums_line, log2_max_p2, n_zp);
          boxcar_.computeSumGpu(nbox, log2_max_p2, n_zp);
          gpu_nbox_path_sum_line = boxcar_.gpuNboxPathSum();
        }

        launchFindTopPathSNRs(gpu_nbox_path_sum_line, rounded_num_timesteps, num_channels,
                              path_offset, drift_block, mu, gpu_sigma_scale_vector, nbox,
                              gpu_top_path_snrs, gpu_top_drift_blocks, gpu_top_path_offsets,
                              gpu_top_path_Nbox);
      }
    }
  }

  cudaMemcpy(cpu_top_path_snrs, gpu_top_path_snrs, num_channels * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_top_drift_blocks, gpu_top_drift_blocks, num_channels * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_top_path_offsets, gpu_top_path_offsets, num_channels * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_top_path_Nbox, gpu_top_path_Nbox, num_channels * sizeof(int),
             cudaMemcpyDeviceToHost);
  checkCuda("dedoppler d->h memcpy");

  double t_dd_sec = (timeInMS() - start_ms) * .001;
  start_ms = timeInMS();

  int window_size = 2 * ceil(normalized_max_drift * drift_timesteps);

  if ((coarse_channel == 0) && (debug >= 1)) {
    printf("foff=%f MHz t_samp=%f sec, n_sti=%d, n_lti=%d, n_avg=%d, n_fft=%d\n",
           metadata.foff * 1e6, metadata.tsamp, n_sti, n_lti, n_avg, num_channels);
    printf("drift_rate_resolution=%.3f drift_timesteps=%d diagonal_drift_rate=%.3f\n",
           drift_rate_resolution, drift_timesteps, diagonal_drift_rate);
    printf("max_drift=%.2f normalized_max_drift=%.2f drift_timesteps=%d window_size=%d=>%.0f Hz\n\n",
           max_drift, normalized_max_drift, drift_timesteps, window_size, window_size * fs);
  }

  const int n_stat_freqs = 20;
  LineStats lstats[n_stat_freqs];

  for (int i = 0; i * window_size < num_channels; ++i) {
    int candidate_freq = -1;
    float candidate_path_snr = snr_threshold;

    for (int j = 0; j < window_size; ++j) {
      int freq = i * window_size + j;
      if (freq >= num_channels) {
        break;
      }
      if (cpu_top_path_snrs[freq] > candidate_path_snr) {
        candidate_freq = freq;
        candidate_path_snr = cpu_top_path_snrs[freq];
      }
    }
    if (candidate_freq < 0) {
      continue;
    }

    int window_end = min(num_channels, candidate_freq + window_size);
    bool found_larger_path_snr = false;
    for (int freq = max(0, candidate_freq - window_size + 1); freq < window_end; ++freq) {
      if (cpu_top_path_snrs[freq] > candidate_path_snr) {
        found_larger_path_snr = true;
        break;
      }
    }
    if (!found_larger_path_snr) {
      int drift_bins = cpu_top_drift_blocks[candidate_freq] * drift_timesteps +
                       cpu_top_path_offsets[candidate_freq];
      double drift_rate = drift_bins * drift_rate_resolution;
      float snr = candidate_path_snr;
      float snr_db = 10 * log10(snr);
      double freq_MHz1 =
          metadata.fch1 + (coarse_channel * num_channels + candidate_freq) * metadata.foff;
      double total_drift_MHz = tsamp * drift_timesteps * drift_rate * 1e-6;
      double freq_MHz2 = freq_MHz1 + total_drift_MHz;
      double freq_MHz_ctr = (freq_MHz1 + freq_MHz2) / 2.;

      int i_subband = candidate_freq / nf_subband;
      int candidate_within_bb_segment = bb_detector.subbandDetected()[i_subband];
      double candidate_blk_sk = blk_sk[i_subband];

      float power = 0.f;
      float drift_tol = .05f;

      bool found_hit = false;

      if ((abs(drift_rate) >= min_drift) && (abs(drift_rate)) <= max_drift + drift_tol) {
        if (do_hit_screen) {
          if (candidate_blk_sk < 3) {
            if (candidate_within_bb_segment) {
              if ((drift_rate < -.6) || (drift_rate > .1)) {
                found_hit = true;
              }
            } else {
              found_hit = true;
            }
          }
        } else {
          found_hit = true;
        }
      }

      if (found_hit) {
        int stamp_boundary_quant = 128 / static_cast<int>(sizeof(float));
        int stamp_width =
            stamp_boundary_quant * (DedopplerConfig::kStampNFreqMax / stamp_boundary_quant);
        int stamp_rows = num_timesteps;
        int stamp_start_column0 = candidate_freq + drift_bins / 2 - stamp_width / 2;
        int stamp_start_column =
            min(num_channels - stamp_width, max(0, stamp_start_column0));
        stamp_start_column = stamp_boundary_quant * (stamp_start_column / stamp_boundary_quant);
        int hit_start_mid = candidate_freq - stamp_start_column;
        int hit_nbox = cpu_top_path_Nbox[candidate_freq];
        float drift_bins_per_line = static_cast<float>(drift_bins) / drift_timesteps;

        stamp_.extractStampGpu(stamp_.gpuStamp(), input.d_sg_data, num_timesteps, num_channels,
                               stamp_start_column, stamp_width, 0, stamp_rows, hit_nbox);
        cudaMemcpy(stamp_.cpuStamp(), stamp_.gpuStamp(),
                   stamp_width * stamp_rows * sizeof(float), cudaMemcpyDeviceToHost);
        checkCuda("cudaMemcpy stamp dev to host");

        float mu_noise = cpu_subband_mean[i_subband];
        float std_noise = cpu_subband_std[i_subband];
        int start_row = 0;
        int start_col = hit_start_mid - 2;

        StampAnalyzer::computeSk(stamp_.cpuStamp(), stamp_rows, stamp_width, mu_noise, std_noise,
                                 n_sti, hit_nbox, start_row, stamp_rows, start_col, n_stat_freqs,
                                 drift_bins_per_line, lstats);

        float hit_sk = lstats[hit_start_mid - start_col].sk;
        float hit_max_min = lstats[hit_start_mid - start_col].max_min_ratio;

        if (do_hit_screen && hit_sk >= 3) {
          found_hit = false;
        }

        if (found_hit) {
          hit_count++;

          if (print_hits) {
            if (hit_count == 1) {
              printf("\n");
            }
            printf("hit %2d: chnl %2d sb %3d %8d %5d %10.3f MHz, %7.3f Hz/sec, SNR %5.2f dB, BlkSK "
                   "%5.2f (%d), Nbox %d, SK %5.3f, maxmin  %5.3f\n",
                   hit_count, coarse_channel, candidate_freq / nf_subband,
                   candidate_freq - num_channels / 2, drift_bins, freq_MHz_ctr, drift_rate, snr_db,
                   candidate_blk_sk, candidate_within_bb_segment, hit_nbox, hit_sk, hit_max_min);
          }

          if (debug >= 3) {
            int hit_end_mid = hit_start_mid + drift_bins;
            int hit_nbox2 = hit_nbox / 2;
            int hit_start_min = hit_start_mid - hit_nbox2;
            int hit_end_max = hit_end_mid - hit_nbox2 + hit_nbox - 1;
            stamp_print_count++;
            if (stamp_print_count <= 10) {
              stamp_.printHitStampDebug(coarse_channel, hit_count, candidate_freq, drift_bins,
                                        stamp_start_column, hit_start_mid, hit_end_mid, hit_nbox,
                                        hit_start_min, hit_end_max, stamp_width, stamp_rows,
                                        drift_bins_per_line, n_stat_freqs, lstats);
            }
          }

          DedopplerHit hit(metadata, candidate_freq, freq_MHz_ctr, freq_MHz1, freq_MHz2, drift_bins,
                           drift_rate, candidate_path_snr, beam, coarse_channel, num_timesteps,
                           power, candidate_blk_sk, hit_sk, hit_max_min);
          output->push_back(hit);
        }
      }
    }
  }

  free(subband_work);

  double t_log_hits_sec = (timeInMS() - start_ms) * .001;
  double t_search_sec = (timeInMS() - start_ms_all) * .001;

  if (debug >= 1) {     
    printf("\nElapsed times: coarse chnl %d, UM %d, fft %d, sti %d, lti %d\n", coarse_channel,
          static_cast<int>(input.managed), num_channels, n_sti, n_lti);
    printf("Input copy:      %.3f sec\n", t_input_copy_sec);
    printf("Sum Columns:     %.3f sec\n", t_sumcols_sec);
    printf("Stats:           %.3f sec\n", t_stats_sec);
    printf("Scale input:     %.3f sec\n", t_scale_sec);
    printf("Taylor GPU:      %.3f sec\n", t_dd_sec);
    printf("Log Hits:        %.3f sec\n", t_log_hits_sec);
    printf("DeDoppler total: %.3f sec\n", t_search_sec);
  }
}
