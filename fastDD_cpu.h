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

typedef struct {
  double f_sg_min;      // SG start freq (bin center) Hz
  double f_sg_max;      // SG start freq (bin center) Hz
  double df_sg;         // SG freq increment Hz
  int Nf_sg;           // SG number of frequency values
  double dt_sg;         // SG time increment per line (time resolution; lines might be averaged)
  int Nt_sg;           // SG number of lines (time values)
  double Lf;            // PFB overlap factor (=1 for FFT filter bank), assumes Lr=Lf 
  int Nt;              // total number of time samples to integrate (Nt/N0 is power of 2)
  int N0;              // first stage number of time samples
  double f_DD_min;      // DD plane start freq (bin center)
  double f_DD_max;      // DD plane end freq (bin center)
  double df_DD;         // DD freq increment
  int Nf_DD;           // DD number of frequency values
  double dfdt_min_nom;  // desired DD minimum frequency rate Hz/sec (input)
  double dfdt_max_nom;  // desired DD maximum frequency rate Hz/sec (input)
  double dfdt_min;      // actual DD minimum frequency rate Hz/sec
  double dfdt_max;      // actual DD maximum frequency rate Hz/sec
  double d_dfdt;        // DD freq rate increment
  double Lr;            // upsample for drift rate increment, Lr=Lf for now
  int Nr;              // final DD number of freq drift rates
  double fs;            // filter bank output sample rate = df_sg*Lf;
  int n_stage;         // number fastDD stages after initial slow stage
  int Nr_ext;          // augmented DD number of freq rates (slightly higher than Nr)
  int Nr0;             // DD number of freq drift rates - first stage
  int N_group0;        // DD number of groups  - first stage
  int m_min0;          // min drift rate index - first stage
  int m_max0;          // max drift rate index - first stage
  int m_min;           // min drift rate index - final stage
  int m_max;           // max drift rate index - final stage
  uint sg_buffer_bytes; // presumed spectrogram size in bytes
  uint DD_buffer_bytes; // minimum DD buffer size in bytes (1 of 2)
  uint total_buffer_bytes; // SG + 2 DD buffer size sum
} DD_metadata;

void gen_fastDD_metadata(DD_metadata *dd,      // DD metadata structure with entries filled 
                          double f_sg_min,      // SG start freq (bin center) Hz
                          double df_sg,         // SG freq increment Hz
                          int Nf_sg,           // SG number of frequency values
                          double dt_sg,         // SG time increment per line (time resolution, lines might be averaged)
                          int Nt_sg,           // SG number of lines (time values)
                          double dfdt_min_nom,  // desired DD minimum frequency rate Hz/sec (input)
                          double dfdt_max_nom,  // desired DD maximum frequency rate Hz/sec (input)
                          double Lf,            // PFB overlap factor (=1 for FFT filter bank) 
                          int Nt,              // total number of time samples to integrate (Nt/N0 is power of 2)
                          int N0);             // first stage number of time samples

void print_fastDD_metadata(DD_metadata *dd);    // DD metadata structure with entries filled 

__global__ void findTopPathSums2(const float* path_sums, int Nr, int num_freqs,
                                int drift_block, float* top_path_sums,
                                int* top_drift_blocks, int* top_path_offsets);

void findTopPathSums_cpu(const float* path_sums, int Nr, int n_freq,
                                int drift_block, float* top_path_sums,
                                int* top_drift_blocks, int* top_path_offsets);

void zeroTopPathSums_cpu( int n_freq, float* top_path_sums,
                          int* top_drift_blocks, int* top_path_offsets);

void fastDD_cpu_stage1(float *det_DD1, float *xx, int m_min0, int m_max0, 
               int Nt, int N0, int n_freq);

void fastDD_cpu_stage_i(float *det_DD_out, float *det_DD_in, int i_stage,
                   int m_min0, int m_max0,int Nt, int N0, int n_freq);

float* fastDD_cpu(float *xx_sg,           // input mag squared SG (spectrogram) values [Nt_sg][Nf_sg]
                  float *det_DD_work[2],  // preallocated work arrays [Nr_ext][Nf_sg] see instructions
                  DD_metadata *dd);       // DD metadata structure with entries filled by gen_fastDD_metadata()

__global__ void gpu_vector_add_xy_z(int n, float * z, float * x, float * y);

__global__ void gpu_vector_add_xy_z1(int n1, int n2, float * z, float * x, float * y);

__global__ void gpu_vector_add_xy_z2(int n1, int n2, float * z, float * x, float * y);

__global__ void gpu_vector_add_xy_x(int n, float * x, float * y);

void fastDD_gpu_stage1(float *det_DD1, float *xx, long m_min0, long m_max0, 
               long Nt, long N0, long n_freq);

void fastDD_gpu_stage_i(float *det_DD_out, float *det_DD_in, int i_stage,
                   int m_min0, int m_max0,int Nt, int N0, int n_freq);

float* fastDD_gpu(float *xx_sg,           // input mag squared SG (spectrogram) values [Nt_sg][Nf_sg]
                  float *det_DD_work[2],  // preallocated work arrays [Nr_ext][Nf_sg] see instructions
                  DD_metadata *dd);       // DD metadata structure with entries filled by gen_fastDD_metadata()
