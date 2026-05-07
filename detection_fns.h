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

void calc_mean_std_dev(const float* x, int n, float *mean, float *std_dev);

void calc_mean_std_dev2(const float* x, int n, float *mean, float *std_dev);

float find_max(const float* x, int n);

float find_min(const float* x, int n);

void calc_subband_mean_std(const float* x_sg, int Nf, int n_subband, bool do_limit, 
                  float *subband_limit, float *work, float *subband_mean, float *subband_std);

void multipass_subband_mean_std(const float* x_sg, int Nf, int n_subband, float shear_constant, 
                      float *work, float *subband_mean, float *subband_std, float *subband_limit);

void DC_replace(float* x, int DC_replace_ofs, int DC_mean_pts);

void print_x_lr(float* x, int max_ofs, float scale);

void print_x_segment(float* x, int n_pts, float scale);

void print_x_segment_stride(float* x, int n_pts, int stride, float scale);

void print_f_x_segment(float* x, int n_pts, float scale, float f0, float df); 

__global__ void gpu_subband_interpolate(float* x, int n_freq, float* x_subband, int n_subband);

__global__ void gpu_zero_mean_unit_std(float* x, float* mu, float * sigma, int n_freq);

__global__ void gpu_local_mean_scale(float* x, float* mu, int n_freq);

__global__ void gpu_compute_sigma_scale(float* sigma_scale, float* sigma, float Nbox_gain, int n_freq);

