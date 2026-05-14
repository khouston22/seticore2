
#include <assert.h>
#include "boxcar_sum.h"

#include <string.h>

void gen_boxcar_p2_sums_cpu(float *p2_path_sums, // output: power of 2 sums array [log2_max_p2+1]*[n_freq+2*n_zp] 
                            float *DD_sums_line, // input: DD sum for single drift value [n_freq]
                            int n_freq,          // number of frequency points in DD_sums_line vector
                            int log2_max_p2,     // log2 of the maximum power of 2 to be calculated
                            int n_zp)            // #zeros padded before and after n_freq spectrum points

{
  // The input is a single row of the DD_sums array corresponding to a single drift value
  // in a drift block, which is indexed by path_offset
  // DD_sums_line[0..n_freq-1] =  taylor_sums[path_offset][0..n_freq-1]
  //
  // This function generates boxcar sum vectors for boxcar widths (Nbox) with power of 2 values,
  // so a boxcar (moving average) filter is computed for each power of 2. 
  // Each boxcar sum applies an FIR filter with an impulse response of Nbox coefficents with value 1,
  // i.e. h(n) = 1 for 0 <= n < Nbox, 0 otherwise
  //
  // The p2_path_sums array is used to efficiently compute boxcar vectors with arbitrary Nbox values
  //
  // Boxcar sums are computed for Nbox = 1, 2, 4, ... Nbox_p2_max
  // where Nbox_p2_max = maximum power of 2 = 2^log2_max_p2
  // e.g. for log2_max_p2=5, Nbox_p2_max = 2^5 = 32
  // The maximum Nbox value that can subsequently calculated is Nbox_max = 2*Nbox_p2_max-1
  // e.g. for log2_max_p2=5 and Nbox_p2_max = 32, we can compute up to Nbox_max = 63
  // 
  // The boxcar sum for a given Nbox = 2^i_Nbox is computed as
  // p2_path_sums[i_Nbox][n] = sum(DD_sums_line[n+m]) for m=0 to Nbox-1
  // over i_Nbox = 0 to log2_max_p2
  // 
  // This function will be called once for every drift value in a drift block
  // or each row of DD_sum array aka the detection plane
  //

  int Nbox_p2_max = 1 << log2_max_p2;
  int Nbox_max = 2*Nbox_p2_max - 1;
  int n_freq_ext = n_freq + 2*n_zp;

  assert(n_zp >= Nbox_max);

  float *p2_row,*p2_row_new;
  
  // zero-pad the end areas 

  for (int i_Nbox=0; i_Nbox<=log2_max_p2; i_Nbox++) {
    p2_row = &p2_path_sums[i_Nbox*n_freq_ext];
    memset(&p2_row[0], 0, n_zp*sizeof(float)); 
    memset(&p2_row[n_freq+n_zp], 0, n_zp*sizeof(float)); 
  }

  // Nbox = 1, do simple copy

  p2_row = &p2_path_sums[0];
  for (int i_freq=n_zp; i_freq<n_freq+n_zp; i_freq++) {
    p2_row[i_freq] = DD_sums_line[i_freq-n_zp];
  }

  // Nbox =2 ...Nbox_p2_max

  int stride = 1;

  for (int i_Nbox=1; i_Nbox<=log2_max_p2; i_Nbox++) {
    p2_row_new = &p2_path_sums[i_Nbox*n_freq_ext];
    p2_row = &p2_path_sums[(i_Nbox-1)*n_freq_ext];
    int Nbox_p2 = 1 << i_Nbox;
  
    for (int i_freq=0; i_freq<n_freq+n_zp+Nbox_p2; i_freq++) {
      p2_row_new[i_freq+stride] = p2_row[i_freq] + p2_row[i_freq+stride]; 
    }  
    stride = 2*stride;
  }

  return;
}

int readBit(int N, int bit_idx) {
  // Reads value of a bit at offset bit_idx in integer N
  return ( (N & (1 << bit_idx)) >> bit_idx );
}

void gen_boxcar_sum_cpu(float *Nbox_path_sum, // output vector for boxcar width Nbox [n_freq] 
                        float *p2_path_sums,  // input: power of 2 sums array [log2_max_p2+1]*[n_freq+2*n_zp]
                        float *work,          // work area [2]*[n_freq+2*n_zp] 
                        int Nbox,             // width of boxcar (number of non-zero points in impulse response)
                        int n_freq,           // number of frequency points in DD sum vector
                        int log2_max_p2,      // log2 of the maximum power of 2 to be calculated
                        int n_zp)             // #zeros padded before and after n_freq spectrum points

{
  // This function generates boxcar sums for a single boxcar width (single value of Nbox)
  // The value of Nbox is general, not limited to a power of 2
  // Nbox can vary from 1 to 2*Nbox_p2_max-1, where Nbox_p2_max = 2^log2_max_p2
  // 
  // The boxcar sum for a given Nbox is equal to
  // work_out[n] = sum(DD_sums_line[n+m]) for m=0 to Nbox-1
  // and the final output is shifted and scaled
  // Nbox_path_sum[n] = work_out[n+Nbox/2]/Nbox
  //
  // Nbox_path_sum is calculated with a fast algorithm which combines the power of 2 path sums
  // and requires at most floor(log2(Nbox))+1 vector summations
  // 
  // This function will be called once for every desired Nbox value in every DD_sums line
  // in a drift block
  //
  // 
 
  assert(Nbox >= 1);
  assert(Nbox < (1 << (log2_max_p2+1)));
  
  int n_freq_ext = n_freq + 2*n_zp;

  /* zero out work array */

  memset(work, 0, 2*n_freq_ext*sizeof(float)); 

  /* bit 0, do simple copy or zeros */

  float *p2_row,*work_in_row,*work_out_row;

  work_out_row = &work[0];
  p2_row = &p2_path_sums[0];
  
  int bit0 = readBit(Nbox,0);
  // printf("Nbox=%d, bit0=%d\n",Nbox,bit0);
  if (bit0) {
    for (int i_freq=0; i_freq<n_freq+2*n_zp; i_freq++) {
      work_out_row[i_freq] = p2_row[i_freq];
    }
  }

  int work_out_idx = 0;
  int work_in_idx  = 1;
  
  /* Sum in power of 2 boxcar sums for Nbox =2 ...Nbox_p2_max */

  for (int i_Nbox=1; i_Nbox<=log2_max_p2; i_Nbox++) {
    int Nbox_p2 = 1 << i_Nbox;
    // printf("Nbox=%d, i_Nbox=%d, bit=%d\n",Nbox,i_Nbox,readBit(Nbox,i_Nbox));
    if (readBit(Nbox,i_Nbox)==1) {
      // if Nbox contains this power of 2
      // toggle indexes
      work_in_idx = work_out_idx;
      work_out_idx = 1 - work_out_idx; 
      // printf("Nbox=%d, i_Nbox=%d, bit=%d, in=%d, out=%d\n",Nbox,i_Nbox,readBit(Nbox,i_Nbox),work_in_idx,work_out_idx);
      work_out_row = &work[work_out_idx*n_freq_ext];
      work_in_row =  &work[work_in_idx*n_freq_ext];
      p2_row = &p2_path_sums[i_Nbox*n_freq_ext];
 
      // sum in power of 2 boxcar sum
      for (int i_freq=Nbox_p2; i_freq<n_freq+n_zp+Nbox_p2; i_freq++) {
        work_out_row[i_freq] = work_in_row[i_freq-Nbox_p2] + p2_row[i_freq]; 
      } 
    }
  }

  /* shift and scale by 1/Nbox */
  int shift = Nbox/2;  // zero freq offset if Nbox odd, half bin if Nbox even
  float scale = 1./Nbox;
  // int shift = 0;  // zero freq offset if Nbox odd, half bin if Nbox even
  // float scale = 1.;
  for (int i_freq=0; i_freq<n_freq; i_freq++) {
    Nbox_path_sum[i_freq] = work_out_row[n_zp+i_freq+shift]*scale; 
  }  

  return;
}

int gen_Nbox_list1(int* Nbox_list, int drift_block, int max_Nbox_bw) 
{
  // generate list of boxcar average Nbox values to evaluate

  int Nbox_drift;
  int n_Nbox;
  
  // First item is just based on the drift block

  if (drift_block >= 0) {
    Nbox_drift = drift_block + 1;
  } else {
    Nbox_drift = -drift_block;
  }
  
  // Nbox_drift = MAX(Nbox_drift,3);   // force min Nbox=3
  // Nbox_drift = 2*(Nbox_drift/2)+1;  // force odd, round up

  Nbox_list[0] = Nbox_drift;
  n_Nbox = 1;

  // Remaining items are powers of 2 (+ 1) beyond 1.5*Nbox_drift (if applicable)

  float min_Nbox_bw = 1.5*Nbox_drift;
    
  if (max_Nbox_bw > min_Nbox_bw) {
    int n_Nbox_bw = (int) floor(log2(MAX(1,max_Nbox_bw)));
    
    int i_bw_min = (int) ceil(log2(min_Nbox_bw));
    if (n_Nbox_bw>=i_bw_min) {
      n_Nbox = n_Nbox_bw - i_bw_min + 2;

      for (int i_bw=i_bw_min; i_bw<=n_Nbox_bw; i_bw++) {
        Nbox_list[i_bw - i_bw_min + 1] = (1 << i_bw)+1; // 2^i_bw + 1
      }
    }
  }
  
  return n_Nbox;
} 

void print_Nbox_list(int* Nbox_list, int n_Nbox, int drift_block) 
{
  // view list for current drift_block index
    
  printf("drift_block=%d, n_Nbox=%d, Nbox = ",drift_block,n_Nbox);
  for (int i_Nbox=0; i_Nbox<n_Nbox; i_Nbox++) printf("%d ",Nbox_list[i_Nbox]);
  printf("\n");

  return;
}

void print_Nbox_segment(float* x, int n_pts, int start_offset, float scale) 
{
  // view vector segment
    
  for (int i_ofs=start_offset; i_ofs<start_offset+n_pts; i_ofs++) {
    if (i_ofs%10==0) printf("\n%6d   ",i_ofs);
    printf("%8.0f ",x[i_ofs]*scale);
  }
  printf("\n");
  return;
} 


void gen_boxcar_p2_sums_gpu(float *gpu_p2_path_sums, // output: power of 2 sums array [log2_max_p2+1]*[n_freq+2*n_zp] 
                            float *gpu_DD_sums_line, // input: DD sum for single drift value [n_freq]
                            int n_freq,              // number of frequency points in DD_sums_line vector
                            int log2_max_p2,         // log2 of the maximum power of 2 to be calculated
                            int n_zp)                // #zeros padded before and after n_freq spectrum points

{
  // The input is a single row of the DD_sums array corresponding to a single drift value
  // in a drift block, which is indexed by path_offset
  // DD_sums_line[0..n_freq-1] =  taylor_sums[path_offset][0..n_freq-1]
  //
  // This function generates boxcar sum vectors for boxcar widths (Nbox) with power of 2 values,
  // so a boxcar (moving average) filter is computed for each power of 2. 
  // Each boxcar sum applies an FIR filter with an impulse response of Nbox coefficents with value 1,
  // i.e. h(n) = 1 for 0 <= n < Nbox, 0 otherwise
  //
  // The p2_path_sums array is used to efficiently compute boxcar vectors with arbitrary Nbox values
  //
  // Boxcar sums are computed for Nbox = 1, 2, 4, ... Nbox_p2_max
  // where Nbox_p2_max = maximum power of 2 = 2^log2_max_p2
  // e.g. for log2_max_p2=5, Nbox_p2_max = 2^5 = 32
  // The maximum Nbox value that can subsequently calculated is Nbox_max = 2*Nbox_p2_max-1
  // e.g. for log2_max_p2=5 and Nbox_p2_max = 32, we can compute up to Nbox_max = 63
  // 
  // The boxcar sum for a given Nbox = 2^i_Nbox is computed as
  // p2_path_sums[i_Nbox][n] = sum(DD_sums_line[n+m]) for m=0 to Nbox-1
  // over i_Nbox = 0 to log2_max_p2
  // 
  // This function will be called once for every drift value in a drift block
  // or each row of DD_sum array aka the detection plane
  //

  int Nbox_p2_max = 1 << log2_max_p2;
  int Nbox_max = 2*Nbox_p2_max - 1;
  int n_freq_ext = n_freq + 2*n_zp;

  int grid_size = (n_freq_ext + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;

  assert(n_zp >= Nbox_max);

  float *p2_row,*p2_row_new;
  
  // zero-pad the end areas 

  for (int i_Nbox=0; i_Nbox<=log2_max_p2; i_Nbox++) {
    p2_row = &gpu_p2_path_sums[i_Nbox*n_freq_ext];
    cudaMemsetAsync(&p2_row[0], 0, n_zp*sizeof(float)); 
    cudaMemsetAsync(&p2_row[n_freq+n_zp], 0, n_zp*sizeof(float)); 
    checkCuda("p2-cudaMemsetAsync");
  }

  // Nbox = 1, do simple copy 

  p2_row = &gpu_p2_path_sums[0];
  // for (int i_freq=n_zp; i_freq<n_freq+n_zp; i_freq++) {
  //   p2_row[i_freq] = DD_sums_line[i_freq-n_zp];
  // }
  cudaMemcpy(&p2_row[n_zp],&gpu_DD_sums_line[0],
              n_freq*sizeof(float), cudaMemcpyDeviceToDevice);
  checkCuda("cudaMemcpy-p2=1");

  // Nbox =2 ...Nbox_p2_max

  int stride = 1;

  for (int i_Nbox=1; i_Nbox<=log2_max_p2; i_Nbox++) {
    p2_row_new = &gpu_p2_path_sums[i_Nbox*n_freq_ext];
    p2_row = &gpu_p2_path_sums[(i_Nbox-1)*n_freq_ext];
    int Nbox_p2 = 1 << i_Nbox;
  
    // for (int i_freq=0; i_freq<n_freq+n_zp+Nbox_p2; i_freq++) {
    //   p2_row_new[i_freq+stride] = p2_row[i_freq] + p2_row[i_freq+stride]; 
    // }  
    gpu_boxcar_add_xy_z<<<grid_size, CUDA_MAX_THREADS>>>
                  (n_freq+n_zp+Nbox_p2,&p2_row_new[stride],&p2_row[0], &p2_row[stride]);
    checkCuda("p2-gpu_boxcar_add_xy_z");
    stride = 2*stride;
  }

  return;
}


/* Simple vector add function 1: Z = X + Y */

__global__ void gpu_boxcar_add_xy_z(int n, float * z, float * x, float * y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) z[i] = x[i] + y[i];
}

/* vector scale function: Z = X*scale, not in place (allows shift)*/

__global__ void gpu_boxcar_scale(int n, float * z, float * x, float scale)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) z[i] = x[i]*scale;
}


void gen_boxcar_sum_gpu(float *gpu_Nbox_path_sum, // output vector for boxcar width Nbox [n_freq] 
                        float *gpu_p2_path_sums,  // input: power of 2 sums array [log2_max_p2+1]*[n_freq+2*n_zp]
                        float *gpu_work,          // work area [2]*[n_freq+2*n_zp] 
                        int Nbox,             // width of boxcar (number of non-zero points in impulse response)
                        int n_freq,           // number of frequency points in DD sum vector
                        int log2_max_p2,      // log2 of the maximum power of 2 to be calculated
                        int n_zp)             // #zeros padded before and after n_freq spectrum points

{
  // This function generates boxcar sums for a single boxcar width (single value of Nbox)
  // The value of Nbox is general, not limited to a power of 2
  // Nbox can vary from 1 to 2*Nbox_p2_max-1, where Nbox_p2_max = 2^log2_max_p2
  // 
  // The boxcar sum for a given Nbox is equal to
  // work_out[n] = sum(DD_sums_line[n+m]) for m=0 to Nbox-1
  // and the final output is shifted and scaled
  // Nbox_path_sum[n] = work_out[n+Nbox/2]/Nbox
  //
  // Nbox_path_sum is calculated with a fast algorithm which combines the power of 2 path sums
  // and requires at most floor(log2(Nbox))+1 vector summations
  // 
  // This function will be called once for every desired Nbox value in every DD_sums line
  // in a drift block
  //
  // 
 
  assert(Nbox >= 1);
  assert(Nbox < (1 << (log2_max_p2+1)));
  
  int n_freq_ext = n_freq + 2*n_zp;

  int grid_size = (n_freq_ext + CUDA_MAX_THREADS - 1) / CUDA_MAX_THREADS;

  /* zero out work array */

  cudaMemsetAsync(gpu_work, 0, 2*n_freq_ext*sizeof(float)); 
  checkCuda("boxcar-cudaMemsetAsync");
  
  /* bit 0, do simple copy or zeros */

  float *p2_row,*work_in_row,*work_out_row;

  work_out_row = &gpu_work[0];
  p2_row = &gpu_p2_path_sums[0];
  
  int bit0 = readBit(Nbox,0);
  // printf("Nbox=%d, bit0=%d\n",Nbox,bit0);
  if (bit0) {
    // for (int i_freq=0; i_freq<n_freq+2*n_zp; i_freq++) {
    //   work_out_row[i_freq] = p2_row[i_freq];
    // }
    cudaMemcpy(&work_out_row[0],&p2_row[0],
              n_freq_ext*sizeof(float), cudaMemcpyDeviceToDevice);
    checkCuda("cudaMemcpy-boxcar");
  }

  int work_out_idx = 0;
  int work_in_idx  = 1;
  
  /* Sum in power of 2 boxcar sums for Nbox =2 ...Nbox_p2_max */

  for (int i_Nbox=1; i_Nbox<=log2_max_p2; i_Nbox++) {
    int Nbox_p2 = 1 << i_Nbox;
  
    if (readBit(Nbox,i_Nbox)==1) {
      // if Nbox contains this power of 2
      // toggle indexes
      work_in_idx = work_out_idx;
      work_out_idx = 1 - work_out_idx; 
      
      work_out_row = &gpu_work[work_out_idx*n_freq_ext];
      work_in_row =  &gpu_work[work_in_idx*n_freq_ext];
      p2_row = &gpu_p2_path_sums[i_Nbox*n_freq_ext];
 
      // sum in power of 2 boxcar sum
      // for (int i_freq=Nbox_p2; i_freq<n_freq+n_zp+Nbox_p2; i_freq++) {
      //   work_out_row[i_freq] = work_in_row[i_freq-Nbox_p2] + p2_row[i_freq]; 
      // } 
      gpu_boxcar_add_xy_z<<<grid_size, CUDA_MAX_THREADS>>>
                    (n_freq+n_zp,&work_out_row[Nbox_p2],&work_in_row[0], &p2_row[Nbox_p2]);
      checkCuda("boxcar-gpu_boxcar_add_xy_z");
    }
  }

  /* shift and scale by 1/Nbox */

  int shift = Nbox/2;  // zero freq offset if Nbox odd, half bin if Nbox even
  float scale = 1./Nbox;

  // for (int i_freq=0; i_freq<n_freq; i_freq++) {
  //   Nbox_path_sum[i_freq] = work_out_row[n_zp+i_freq+shift]*scale; 
  // }  

  gpu_boxcar_scale<<<grid_size, CUDA_MAX_THREADS>>>
                (n_freq,&gpu_Nbox_path_sum[0],&work_out_row[n_zp+shift],scale);
  checkCuda("boxcar-gpu_boxcar_scale");

  return;
}


