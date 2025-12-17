#pragma once

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "cuda_util.h"

#if !defined(MAX)
#define	MAX(A, B)	(((A) > (B)) ? (A) : (B))
#endif

#if !defined(MIN)
#define	MIN(A, B)	(((A) < (B)) ? (A) : (B))
#endif

#define DC_REPLACE_ENABLE 1
#define DC_MEAN_PTS 40
#define DC_REPLACE_OFS 15
  
#define N_SUBBAND 128
#define N_SUBBAND_MAX 256
// Minimum number of freq bins per subband - for low SNR variability
#define NF_SUBBAND_MIN 8192

void sumColumns_cpu(const float* input, float* sums, int num_timesteps, int n_freq);

void calc_mean_std_dev(const float* x, int n, float *mean, float *std_dev);

void calc_mean_std_dev2(const float* x, int n, float *mean, float *std_dev);

void calc_subband_mean_std(const float* x_sg, int Nf, int n_subband, bool do_limit, 
                  float *subband_limit, float *work, float *subband_mean, float *subband_std);

void multipass_subband_mean_std(const float* x_sg, int Nf, int n_subband, float shear_constant, 
                      float *work, float *subband_mean, float *subband_std, float *subband_limit);

void ncoh_avg_test(const float* x, int Nf, int Nt, int n_sti, int n_subband);

void DC_replace(float* x, int DC_replace_ofs, int DC_mean_pts);

void print_x_lr(float* x, int max_ofs, float scale);

void print_x_segment(float* x, int n_pts, float scale);

void print_x_segment_stride(float* x, int n_pts, int stride, float scale);

__global__ void gpu_subband_interpolate(float* x, int n_freq, float* x_subband, int n_subband);

__global__ void gpu_zero_mean_unit_std(float* x, float* mu, float * sigma, int n_freq);

__global__ void gpu_local_mean_scale(float* x, float* mu, int n_freq);

__global__ void gpu_compute_sigma_scale(float* sigma_scale, float* sigma, float sqrtNbox, int n_freq);

