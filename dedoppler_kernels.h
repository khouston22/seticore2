#pragma once

#include "cuda_util.h"

void launchSumColumns(const float* input, float* sums, int num_timesteps, int num_freqs, float scale);

void launchFindTopPathSNRs(const float* path_sums_line, int num_timesteps, int num_freqs,
                           int path_offset, int drift_block, float mu, float* sigma_scale,
                           int nbox, float* top_path_snrs, int* top_drift_blocks,
                           int* top_path_offsets, int* top_path_nbox);

void sumColumnsCpu(const float* input, float* sums, int num_timesteps, int n_freq);
