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

void sumColumns_cpu(const float* input, float* sums, int num_timesteps, int n_freq);

void calc_mean_std_dev(const float* x, int n, float *mean, float *std_dev);

void calc_mean_std_dev2(const float* x, int n, float *mean, float *std_dev);

void calc_subband_mean_std(const float* x_sg, int Nf, int n_subband, bool do_limit, 
                  float *subband_limit, float *work, float *subband_mean, float *subband_std);

void multipass_subband_mean_std(const float* x_sg, int Nf, int n_subband, float shear_constant, 
                      float *work, float *subband_mean, float *subband_std, float *subband_limit);

void ncoh_avg_test(const float* x, int Nf, int Nt, int n_sti, int n_subband);
