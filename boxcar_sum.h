#pragma once

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include "cuda_util.h"

#if !defined(MAX)
#define	MAX(A, B)	(((A) > (B)) ? (A) : (B))
#endif

#if !defined(MIN)
#define	MIN(A, B)	(((A) < (B)) ? (A) : (B))
#endif

void gen_boxcar_p2_sums_cpu(float *DD_sums_line, // input: DD sum for single drift value [n_freq]
                            float *p2_path_sums, // output: power of 2 sums array [log2_max_p2+1]*[n_freq+2*n_zp] 
                            int n_freq,          // number of frequency points in path_sum_line vector
                            int log2_max_p2,     // log2 of the maximum power of 2 to be calculated
                            int n_zp);           // #zeros padded before and after n_freq spectrum points


void gen_boxcar_sum_cpu(float *p2_path_sums,  // input: power of 2 sums array [log2_max_p2+1]*[n_freq+2*n_zp]
                        float *work,          // work area [2]*[n_freq+2*n_zp] 
                        float *Nbox_path_sum, // output vector for boxcar width Nbox [n_freq] 
                        int Nbox,             // width of boxcar (number of non-zero points in impulse response)
                        int n_freq,           // number of frequency points in DD sum vector
                        int log2_max_p2,      // log2 of the maximum power of 2 to be calculated
                        int n_zp);            // #zeros padded before and after n_freq spectrum points

void print_Nbox_segment(float* x, int n_pts, float scale);

void gen_boxcar_p2_sums_gpu(float *gpu_DD_sums_line, // input: DD sum for single drift value [n_freq]
                            float *gpu_p2_path_sums, // output: power of 2 sums array [log2_max_p2+1]*[n_freq+2*n_zp] 
                            int n_freq,              // number of frequency points in path_sum_line vector
                            int log2_max_p2,         // log2 of the maximum power of 2 to be calculated
                            int n_zp);               // #zeros padded before and after n_freq spectrum points

void gen_boxcar_sum_gpu(float *gpu_p2_path_sums,  // input: power of 2 sums array [log2_max_p2+1]*[n_freq+2*n_zp]
                        float *gpu_work,          // work area [2]*[n_freq+2*n_zp] 
                        float *gpu_Nbox_path_sum, // output vector for boxcar width Nbox [n_freq] 
                        int Nbox,             // width of boxcar (number of non-zero points in impulse response)
                        int n_freq,           // number of frequency points in DD sum vector
                        int log2_max_p2,      // log2 of the maximum power of 2 to be calculated
                        int n_zp);            // #zeros padded before and after n_freq spectrum points

__global__ void gpu_boxcar_add_xy_z(int n, float * z, float * x, float * y);

__global__ void gpu_boxcar_scale(int n, float * z, float * x, float scale);


